import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import requests
import openai
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from Bio import Entrez

Image.MAX_IMAGE_PIXELS = None

# 세션 상태 초기화
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# LLM RAG 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 난소암 병리 이미지를 해석하는 AI 전문가입니다.

다음은 논문에서 발췌한 내용입니다:
---------------------
{context}
---------------------

이 내용을 바탕으로 아래 질문에 답변하세요.
질문: {question}
"""
)

# PubMed Boolean Query 생성
def generate_pubmed_query_from_question(question: str, llm=None) -> str:
    if llm is None:
        llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=st.session_state.get("user_api_key", ""))
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""아래 설명은 병리 이미지 AI 분석 결과입니다.
이 설명에 맞는 PubMed 검색 Boolean 쿼리를 생성하세요.

설명:
{question}

출력 형식: Boolean Query
"""
    )
    return llm.invoke(query_prompt.format(question=question)).content.strip()

# PubMed 검색
def search_pubmed(question: str, max_results: int = 3):
    Entrez.email = "teamovianai@gmail.com"
    query = question  # LLM Boolean query 안 쓰고 그냥 질문 그대로 사용
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    summaries = []

    for pmid in ids:
        summary = Entrez.esummary(db="pubmed", id=pmid, retmode="xml")
        summary_record = Entrez.read(summary)
        title = summary_record[0].get("Title", "[제목 없음]")
        pubdate = summary_record[0].get("PubDate", "")
        year = pubdate.split()[0] if pubdate else ""
        authors = summary_record[0].get("AuthorList", [])
        author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract = fetch.read()

        summaries.append(Document(
            page_content=abstract,
            metadata={"pmid": pmid, "title": title, "year": year, "authors": author_str, "source": "pubmed"}
        ))
    return summaries

# 이미지 기반 논문 검색 (예시 ViT 사용)
def find_similar_papers_from_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return [
        Document(page_content="Ovarian cancer tissue with high mitotic index...", metadata={"pmid": "12345678", "source": "pubmed"}),
        Document(page_content="Study on serous subtype of ovarian carcinoma...", metadata={"pmid": "87654321", "source": "pubmed"})
    ]

# ✅ Streamlit UI
st.set_page_config(page_title="Ovarian Cancer RAG", layout="wide")

st.sidebar.title("🔐 OpenAI API KEY 입력")
user_api_key = st.sidebar.text_input("API Key", type="password")
if user_api_key:
    st.session_state["user_api_key"] = user_api_key
openai.api_key = st.session_state.get("user_api_key", "")

st.title("🔬 난소암 분석 AI 어시스턴트")

question = st.text_input("텍스트 질문 입력", placeholder="예시: What are the subtypes of ovarian cancer?")
uploaded_file = st.file_uploader("조직 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if st.button("분석 실행"):
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=st.session_state.get("user_api_key", ""))
    embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.1")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(Image.open(BytesIO(image_bytes)), caption="업로드된 조직 이미지", use_container_width=True)

        try:
            with st.spinner("1️⃣ 이미지 업로드 및 Flask 전송 중..."):
                files = {'file': (uploaded_file.name, image_bytes)}
                response = requests.post("http://localhost:5001/infer", files=files)

            if response.status_code == 200:
                result = response.json()
                pred_class = result["pred_class"]
                softmax_probs = result["softmax_probs"]
                attention_base64 = result["attention_map_base64"]

                st.success("✅ Flask 서버 추론 완료!")

                with st.spinner("2️⃣ Attention Map 시각화 중..."):
                    st.image(BytesIO(base64.b64decode(attention_base64)), caption="AI Attention Map", use_container_width=True)

                label_dict = {0: 'HGSC', 1: 'LGSC', 2: 'CC', 3: 'EC', 4: 'MC'}

                st.success(f"✅ AI 예측 클래스: {pred_class} ({label_dict[pred_class]})")

                probs_percent = [
                    f"{label_dict[i]}: {int(p * 100)}%"
                    for i, p in enumerate(softmax_probs)
                ]

                st.write("Softmax Probabilities:")
                for line in probs_percent:
                    st.write(f"- {line}")

                with st.spinner("3️⃣ PubMed 관련 논문 검색 중..."):
                    ai_description = f"Predicted class: {label_dict[pred_class]}, softmax: {probs_percent}"
                    related_papers = search_pubmed(ai_description, max_results=3)

                st.success("✅ PubMed 검색 완료!")
                # ✅ uploads 폴더 비우기
                try:
                    clear_response = requests.post("http://localhost:5001/clear_uploads")
                    if clear_response.status_code == 200:
                        st.success("✅ 서버 uploads 폴더 정리 완료!")
                    else:
                        st.warning(f"⚠️ uploads 폴더 정리 실패: {clear_response.text}")
                except Exception as e:
                    st.warning(f"⚠️ Flask 서버 정리 요청 실패: {e}")

                

                st.subheader("📄 PubMed 관련 논문")
                for doc in related_papers:
                    title = doc.metadata.get("title", "[제목 없음]")
                    year = doc.metadata.get("year", "")
                    authors = doc.metadata.get("authors", "")
                    pmid = doc.metadata.get("pmid", "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                    st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

            else:
                st.error(f"❌ Flask 추론 에러: {response.text}")


        except Exception as e:
            st.error(f"🚨 Flask 서버 연결 실패: {e}")

    elif question:
        docs = search_pubmed(question=question)
        splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        result = rag_chain.invoke({"query": question})

        st.session_state.qa_history.insert(0, {
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"]
        })

        st.subheader("🖍️ 요약 답변")
        st.write(result["result"])
        st.subheader("📄 논문 출처")
        for doc in result["source_documents"]:
            pmid = doc.metadata.get("pmid")
            title = doc.metadata.get("title", "[제목 없음]")
            year = doc.metadata.get("year", "")
            authors = doc.metadata.get("authors", "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

# ✅ Q&A 히스토리 표시
if len(st.session_state.qa_history) > 1:
    st.markdown("## 📚 이전 Q&A 기록")
    for idx, entry in enumerate(st.session_state.qa_history[1:]):
        with st.expander(f"Q{len(st.session_state.qa_history) - idx - 1}: {entry['question']}"):
            st.write(entry["answer"])
            for doc in entry["sources"]:
                pmid = doc.metadata.get("pmid")
                title = doc.metadata.get("title", "[제목 없음]")
                year = doc.metadata.get("year", "")
                authors = doc.metadata.get("authors", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

# ✅ 초기화 버튼
if st.button("기록 초기화"):
    st.session_state.qa_history = []
    st.rerun()
