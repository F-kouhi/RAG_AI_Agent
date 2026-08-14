"""
اپلیکیشن رابط کاربری سیستم RAG فارسی
=====================================
این فایل یک رابط کاربری جامع با Streamlit برای سیستم RAG آموزشی فراهم می‌کند.

ماژول‌های استفاده شده:
- pdf_to_txt: پردازش فایل‌های PDF
- build_embeddings: ساخت embeddings از اسناد
- embeddings: تولید امبدینگ متن
- retriever: بازیابی اسناد با FAISS
- llm_wrapper: پیچیدن مدل زبانی بزرگ
- rag_agent: عامل RAG برای پاسخ به سؤالات
"""

import streamlit as st
import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

# تنظیمات صفحه
st.set_page_config(
    page_title="سیستم RAG فارسی",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل سفارشی برای راست‌چین
st.markdown("""
<style>
    /* راست‌چین کردن کل صفحه */
    .stApp {
        direction: rtl;
        text-align: right;
    }
    /* هدر و عناوین */
    h1, h2, h3, h4, h5, h6 {
        text-align: right;
    }
    /* نوار کناری */
    [data-testid="StSidebar"] {
        direction: rtl;
    }
    /* کارت‌های اطلاعات */
    .info-card {
        
        padding: 1rem;
        border-radius: 0.5rem;
        border-right: 4px solid #4A90D9;
        margin-bottom: 1rem;
    }
    /* کارت پاسخ */
    .answer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    /* کارت چانک‌ها */
    .chunk-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    /* دکمه اصلی */
    .stButton>button {
        direction: rtl;
        width: 100%;
    }
    /* فیلدهای ورودی */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        direction: rtl;
        text-align: right;
    }
    /* انتخابگرها */
    .stSelectbox>div>div>select {
        direction: rtl;
        text-align: right;
    }
    /* آپلود فایل */
    .stFileUploader>div>div>button {
        direction: rtl;
    }
    /* جدول‌ها */
    table {
        direction: rtl;
    }
    /* ناوبری */
    .nav-item {
        padding: 0.8rem 1rem;
        margin: 0.2rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .nav-item:hover {
        background-color: rgba(74, 144, 217, 0.1);
    }
    .nav-item.active {
        background-color: rgba(74, 144, 217, 0.2);
        font-weight: bold;
    }
    /* نوار پیشرفت */
    .progress-container {
        margin: 1rem 0;
    }
    /* تایم‌لاین تاریخچه */
    .history-item {
        padding: 0.8rem;
        border-bottom: 1px solid #eee;
        margin-bottom: 0.5rem;
    }


    /* فیکس slider در حالت RTL */
[data-testid="stSlider"] {
    direction: ltr !important;
}

/* راست‌چین کردن label */
[data-testid="stSlider"] label {
    direction: rtl !important;
    text-align: right !important;
}

/* راست‌چین کردن help text */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    direction: rtl !important;
    text-align: right !important;
}

/* فیکس مقدار نمایشی slider */
[data-testid="stSlider"] [data-testid="stTickBar"] {
    direction: ltr !important;
}
            
/* مخفی کردن کامل sidebar در حالت collapsed */
[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0 !important;
    min-width: 0 !important;
    margin-right: 0 !important;
}

/* مخفی کردن محتوای sidebar در حالت collapsed */
[data-testid="stSidebar"][aria-expanded="false"] > div {
    display: none !important;
}

/* دکمه باز/بسته کردن sidebar */
[data-testid="collapsedControl"] {
    right: auto !important;
    left: 1rem !important;
}

/* فیکس slider در RTL */
[data-testid="stSlider"] {
    direction: ltr !important;
}

[data-testid="stSlider"] label {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    direction: rtl !important;
    text-align: right !important;
}

/* فیکس دکمه‌ها در sidebar */
[data-testid="stSidebar"] button {
    width: 100%;
    text-align: right;
    direction: rtl;
}

/* فیکس فاصله‌گذاری */
[data-testid="stSidebar"] .element-container {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# کلاس‌ها و توابع ماژول - Import از ماژول‌های پروژه
# ============================================================================

from pdf_to_txt import DockumentProcessor
from build_embeddings import EmbeddingBuilder
from embeddings import Embedder
from retriever import Retriever
from llm_wrapper import OpenRouterLLM
from rag_agent import RAGAgent
from hazm import sent_tokenize
import numpy as np


# ============================================================================
# مسیرهای فایل‌ها
# ============================================================================

DATA_DIR = Path("Data")
CHUNKS_PATH = DATA_DIR / "chunks.json"
METADATA_PATH = DATA_DIR / "metadata.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
DOC_PATH = DATA_DIR


# ============================================================================
# توابع کمکی
# ============================================================================

def init_session_state():
    """راه‌اندازی وضعیت جلسه (session state)"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "embedding_model": "intfloat/multilingual-e5-small",
            "embedding_type": "api",
            "top_k": 5,
            "temperature": 0.2,
            "max_tokens": 1024,
            "llm_model": "gapgpt-qwen-3.5",
            "api_url": "https://api.gapgpt.app/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "system_prompt": "شما یک دستیار فارسی هستید که پاسخ‌ها را به صورت خلاصه اما دقیق ارائه می‌دهد."
        }
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None
    if "processing_progress" not in st.session_state:
        st.session_state.processing_progress = 0


def get_settings_path() -> Path:
    """مسیر فایل ذخیره تنظیمات"""
    return DATA_DIR / "app_settings.json"


def save_settings():
    """ذخیره تنظیمات در فایل"""
    settings_path = get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.settings, f, ensure_ascii=False, indent=2)
    st.success("تنظیمات با موفقیت ذخیره شد! ✅")


def load_settings():
    """بارگذاری تنظیمات از فایل"""
    settings_path = get_settings_path()
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            st.session_state.settings.update(loaded)
            st.success("تنظیمات با موفقیت بارگذاری شد! ✅")
    else:
        st.info("فایل تنظیمات یافت نشد. از تنظیمات پیش‌فرض استفاده می‌شود.")


def init_rag_agent():
    """راه‌اندازی عامل RAG با تنظیمات فعلی"""
    try:
        # بررسی وجود embeddings
        if not all(p.exists() for p in [CHUNKS_PATH, METADATA_PATH, EMBEDDINGS_PATH]):
            st.error("❌ فایل‌های embeddings یافت نشد. لطفاً ابتدا اسناد را آپلود و پردازش کنید.")
            return None

        # ایجاد retriever
        embedder_device = "cpu"  # FAISS روی CPU کار می‌کند
        retriever = Retriever(
            chunks_path=str(CHUNKS_PATH),
            metadata_path=str(METADATA_PATH),
            embeddings_path=str(EMBEDDINGS_PATH),
            device=embedder_device
        )
        st.session_state.retriever = retriever

        # ایجاد LLM
        settings = st.session_state.settings
        llm = OpenRouterLLM(
            api_key=settings["api_key"],
            model=settings["llm_model"],
            base_url=settings["api_url"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            system_prompt=settings["system_prompt"]
        )
        st.session_state.llm = llm

        # ایجاد RAG Agent
        user_prompt = (
            "از متن‌های زیر برای پاسخ دقیق به سؤال استفاده کن. "
            "اگر پاسخ در متن‌ها نبود، صادقانه بگو که اطلاعات کافی وجود ندارد. "
            "در پایان هر پاسخ، از کاربر بپرس آیا سؤال دیگری دارد یا می‌خواهد ادامه دهد."
        )
        agent = RAGAgent(retriever, llm, user_prompt=user_prompt)
        st.session_state.rag_agent = agent

        return agent
    except Exception as e:
        st.error(f"❌ خطا در راه‌اندازی RAG: {str(e)}")
        return None


def process_pdf_file(pdf_file: any) -> tuple[bool, str]:
    """
    پردازش فایل PDF و ساخت embeddings
    بازگشت: (موفقیت، پیام)
    """
    try:
        # ذخ موقت فایل
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(DATA_DIR)) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_pdf_path = Path(tmp_file.name)

        doc_id = tmp_pdf_path.stem
        processed_path = DATA_DIR / f"processed_{doc_id}.json"

        # مرحله 1: تبدیل PDF به متن
        st.info("⏳ مرحله 1: تبدیل PDF به متن...")
        processor = DockumentProcessor(
            pdf_path=tmp_pdf_path,
            output_path=processed_path
        )
        processor.pdf_to_json()

        # خواندن صفحات پردازش شده
        pages = json.loads(processed_path.read_text(encoding="utf-8"))
        num_pages = len(pages)
        st.success(f"✅ {num_pages} صفحه پردازش شد.")

        # مرحله 2: chunking و embedding
        st.info("⏳ مرحله 2: ساخت chunks و embeddings...")

        # بررسی فایل‌های موجود
        existing_chunks = []
        existing_metadata = []
        existing_embeddings = None

        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        if EMBEDDINGS_PATH.exists():
            existing_embeddings = np.load(EMBEDDINGS_PATH)

        # chunking
        chunk_size = 700
        overlap_sentences = 2
        all_chunks = existing_chunks.copy()
        all_metadata = existing_metadata.copy()
        all_embeddings_list = []

        if existing_embeddings is not None:
            all_embeddings_list.append(existing_embeddings)

        for page in pages:
            sentences = sent_tokenize(page['text'])
            current_chunk = []
            char_start = 0

            for sent in sentences:
                current_chunk.append(sent)
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)

                if current_len >= chunk_size:
                    chunk_text = " ".join(current_chunk)
                    all_chunks.append(chunk_text)
                    all_metadata.append({
                        "doc_id": doc_id,
                        "char_start": char_start,
                        "char_end": char_start + len(chunk_text),
                        "sentences": current_chunk.copy(),
                        "page_number": page["page_number"]
                    })

                    overlap_sents = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    char_start += len(" ".join(current_chunk[:-overlap_sentences])) if overlap_sentences > 0 else len(chunk_text)
                    current_chunk = overlap_sents

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                all_chunks.append(chunk_text)
                all_metadata.append({
                    "doc_id": doc_id,
                    "char_start": char_start,
                    "char_end": char_start + len(chunk_text),
                    "sentences": current_chunk.copy(),
                    "page_number": page["page_number"]
                })

        # embedding
        embedder = Embedder(
            embedder_type=st.session_state.settings["embedding_type"],
            model_name=st.session_state.settings["embedding_model"]
        )

        batch_size = 64
        new_embeddings = []
        new_chunks = all_chunks[len(existing_chunks):]

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            emb = embedder.embed_text_api(batch)
            new_embeddings.append(emb)
            progress = min(100, int(((i + batch_size) / len(new_chunks)) * 80))
            st.session_state.processing_progress = progress
            st.progress(progress / 100)

        if new_embeddings:
            all_embeddings_list.append(np.vstack(new_embeddings))

        # ذخیره نتایج
        if all_embeddings_list:
            final_embeddings = np.vstack(all_embeddings_list)
        else:
            final_embeddings = np.array([])

        np.save(EMBEDDINGS_PATH, final_embeddings)

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False)

        # پاک‌سازی فایل موقت
        tmp_pdf_path.unlink(missing_ok=True)

        st.success(f"✅ پردازش کامل شد! تعداد کل chunks: {len(all_chunks)}")
        st.session_state.processing_progress = 100
        st.session_state.upload_status = {
            "success": True,
            "total_chunks": len(all_chunks),
            "total_pages": num_pages,
            "doc_id": doc_id
        }
        return True, f"پردازش کامل شد! {len(all_chunks)} chunk از {num_pages} صفحه."

    except Exception as e:
        st.error(f"❌ خطا در پردازش: {str(e)}")
        return False, str(e)


# ============================================================================
# صفحه اصلی - خانه
# ============================================================================

def render_home_page():
    """صفحه اصلی - معرفی سیستم RAG"""
    st.markdown("# 📚 سیستم RAG فارسی")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h2>🤖 دستیار هوشمند پرسش و پاسخ</h2>
            <p style="font-size: 1.1rem;">
                این سیستم با استفاده از فناوری <strong>RAG (Retrieval-Augmented Generation)</strong> 
                امکان پرسش و پاسخ از اسناد فارسی را فراهم می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## ✨ ویژگی‌های سیستم")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📄 پردازش اسناد</h3>
            <ul>
                <li>آپلود فایل PDF</li>
                <li>تبدیل خودکار به متن</li>
                <li>تمیزسازی متن</li>
                <li>تقسیم به بخش‌های کوچک</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 بازیابی هوشمند</h3>
            <ul>
                <li>امبدینگ پیشرفته</li>
                <li>جستجوی FAISS</li>
                <li>رتبه‌بندی مرتبط بودن</li>
                <li>بهینه‌سازی نتایج</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>💬 پاسخ‌دهی هوشمند</h3>
            <ul>
                <li>مدل زبانی بزرگ</li>
                <li>پاسخ به فارسی</li>
                <li>منبع‌دهی دقیق</li>
                <li>تاریخچه مکالمه</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # نمایش وضعیت سیستم
    st.markdown("## 📊 وضعیت سیستم")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            st.metric("تعداد Chunks", len(chunks))
        else:
            st.metric("تعداد Chunks", 0)
    
    with col2:
        if EMBEDDINGS_PATH.exists():
            emb = np.load(EMBEDDINGS_PATH)
            st.metric("ابعاد امبدینگ", emb.shape[1])
        else:
            st.metric("ابعاد امبدینگ", "-")
    
    with col3:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            doc_ids = set(m.get("doc_id", "") for m in metadata)
            st.metric("تعداد اسناد", len(doc_ids))
        else:
            st.metric("تعداد اسناد", 0)
    
    with col4:
        agent_ready = st.session_state.rag_agent is not None
        status_icon = "✅" if agent_ready else "❌"
        status_text = "آماده" if agent_ready else "غیرفعال"
        st.metric("وضعیت عامل", f"{status_icon} {status_text}")


# ============================================================================
# صفحه آپلود و پردازش اسناد
# ============================================================================

def render_upload_page():
    """صفحه آپلود و پردازش اسناد"""
    st.markdown("# 📤 آپلود و پردازش اسناد")
    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h3>📖 راهنمای آپلود</h3>
        <p>فایل‌های PDF خود را آپلود کنید تا سیستم به صورت خودکار آن‌ها را پردازش کند:</p>
        <ol>
            <li>فایل‌های PDF را انتخاب یا بکشید و رها کنید (آپلود همزمان چند فایل)</li>
            <li>سیستم متن را استخراج و تمیزسازی می‌کند</li>
            <li>متن به بخش‌های کوچک (Chunk) تقسیم می‌شود</li>
            <li>برای هر بخش امبدینگ ساخته می‌شود</li>
            <li>نتایج برای پرسش و پاسخ آماده می‌شود</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # آپلود فایل (چند فایل همزمان)
    uploaded_files = st.file_uploader(
        "فایل‌های PDF خود را انتخاب کنید",
        type=["pdf"],
        accept_multiple_files=True,
        help="حداکثر حجم هر فایل: 50 مگابایت - می‌توانید چند فایل همزمان آپلود کنید",
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.info(f"📄 تعداد {len(uploaded_files)} فایل انتخاب شده: {', '.join(f.name for f in uploaded_files)}")

        # بررسی تکراری بودن
        existing_chunks_count = 0
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                existing_chunks_count = len(json.load(f))

        if st.button("🚀 شروع پردازش همه فایل‌ها", type="primary"):
            progress_container = st.container()
            
            with progress_container:
                total_files = len(uploaded_files)
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                # جدول وضعیت پردازش فایل‌ها
                processed_results = []
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.info(f"⏳ در حال پردازش فایل {i+1} از {total_files}: {uploaded_file.name}...")
                    
                    success, message = process_pdf_file(uploaded_file)
                    processed_results.append({
                        "name": uploaded_file.name,
                        "success": success,
                        "message": message
                    })
                    
                    overall_progress.progress((i + 1) / total_files)

                # نمایش نتایج پردازش
                status_text.empty()
                st.markdown("---")
                st.markdown("## 📋 نتایج پردازش فایل‌ها")
                
                for result in processed_results:
                    if result["success"]:
                        st.success(f"✅ {result['name']}: {result['message']}")
                    else:
                        st.error(f"❌ {result['name']}: {result['message']}")
                
                # بازنشانی عامل RAG پس از پردازش
                st.session_state.rag_agent = None
                st.session_state.retriever = None
                st.session_state.llm = None
                st.rerun()

    # نمایش وضعیت پردازش قبلی
    st.markdown("---")
    st.markdown("## 📋 وضعیت فایل‌های پردازش شده")

    if st.session_state.upload_status:
        status = st.session_state.upload_status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("وضعیت", "✅ موفق")
        with col2:
            st.metric("تعداد Chunks", status["total_chunks"])
        with col3:
            st.metric("تعداد صفحات", status["total_pages"])
    else:
        st.info("هنوز فایلی پردازش نشده است.")

    # نمایش اطلاعات فایل‌ها
    st.markdown("---")
    st.markdown("## 🗂️ فایل‌های داده")

    col1, col2, col3 = st.columns(3)
    with col1:
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            size_mb = CHUNKS_PATH.stat().st_size / (1024 * 1024)
            st.success(f"✅ chunks.json\n{len(chunks)} chunk\n{size_mb:.2f} MB")
        else:
            st.warning("❌ chunks.json یافت نشد")

    with col2:
        if METADATA_PATH.exists():
            size_mb = METADATA_PATH.stat().st_size / (1024 * 1024)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            doc_ids = set(m.get("doc_id", "") for m in metadata)
            st.success(f"✅ metadata.json\n{len(metadata)} ردیف\n{size_mb:.2f} MB")
        else:
            st.warning("❌ metadata.json یافت نشد")

    with col3:
        if EMBEDDINGS_PATH.exists():
            size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
            st.success(f"✅ embeddings.npy\n{size_mb:.2f} MB")
        else:
            st.warning("❌ embeddings.npy یافت نشد")

    # دکمه حذف داده‌ها
    st.markdown("---")
    col_center1, col_center2 = st.columns([2, 1])
    with col_center1:
        if st.button("🗑️ حذف تمام داده‌ها و شروع مجدد", type="secondary", key="delete_data"):
            st.session_state.upload_status = None
            st.session_state.processing_progress = 0
            st.warning("⚠️ آیا مطمئن هستید؟")


# ============================================================================
# صفحه پرسش و پاسخ
# ============================================================================

def render_qa_page():
    """صفحه پرسش و پاسخ"""
    st.markdown("# 💬 پرسش و پاسخ")
    st.markdown("---")

    # بررسی آماده بودن عامل
    if st.session_state.rag_agent is None:
        agent_ready = (CHUNKS_PATH.exists() and 
                       METADATA_PATH.exists() and 
                       EMBEDDINGS_PATH.exists())
        
        if not agent_ready:
            st.error("❌ داده‌های لازم برای پرسش و پاسخ یافت نشد.")
            st.info("⬅️ لطفاً ابتدا از صفحه «آپلود اسناد» فایل‌های لازم را پردازش کنید.")
            return
        else:
            with st.spinner("در حال راه‌اندازی عامل RAG..."):
                init_rag_agent()

    if st.session_state.rag_agent is None:
        st.error("❌ خطا در راه‌اندازی عامل RAG. لطفاً صفحه را تازه‌سازی کنید.")
        return

    st.success("✅ عامل RAG آماده پاسخ‌دهی است.")

    # ورودی سؤال
    st.markdown("## ❓ سؤال خود را بپرسید")

    query = st.text_area(
        "سؤال خود را به زبان فارسی بنویسید:",
        placeholder="مثال: ثبت‌نام دانشجویان مهمان چگونه انجام می‌شود؟",
        height=80
    )

    # دکمه ارسال
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        send_btn = st.button("🚀 ارسال سؤال", type="primary", key="send_query")

    if send_btn and query.strip():
        top_k = st.session_state.settings["top_k"]

        with st.spinner("⏳ در حال جستجو و تولید پاسخ..."):
            start_time = time.time()
            
            try:
                results = st.session_state.retriever.retrieve(query, top_k=top_k)
                context_blocks = []
                for i, r in enumerate(results, 1):
                    context_blocks.append(f"[{i}] {r['text']}")
                context_text = "\n\n".join(context_blocks)

                prompt = f"""
{st.session_state.rag_agent.user_prompt}
متن‌ها:
{context_text}

سؤال:
{query}
"""
                answer = st.session_state.llm.generate(prompt)
                elapsed = time.time() - start_time

                # ذخیره در تاریخچه
                st.session_state.conversation_history.insert(0, {
                    "query": query,
                    "answer": answer,
                    "chunks": results,
                    "time": time.strftime("%H:%M"),
                    "elapsed": elapsed
                })

                # نمایش پاسخ
                st.markdown(f"### 💡 پاسخ (زمان: {elapsed:.1f} ثانیه)")
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                # نمایش chunks مرتبط
                st.markdown(f"### 📑 {len(results)} قطعه مرتبط بازیابی شده:")

                for i, chunk in enumerate(results):
                    with st.expander(f"📄 قطعه {i+1} (امتیاز: {chunk['score']:.3f})", expanded=(i == 0)):
                        st.markdown(chunk["text"])
                        st.caption(f"امتیاز شباهت: {chunk['score']:.4f}")
                        if "page_number" in chunk["metadata"]:
                            st.caption(f"صفحه: {chunk['metadata']['page_number']}")

            except Exception as e:
                st.error(f"❌ خطا در تولید پاسخ: {str(e)}")

    elif send_btn and not query.strip():
        st.warning("⚠️ لطفاً سؤال خود را بنویسید.")

    # تاریخچه مکالمات
    st.markdown("---")
    st.markdown("## 📜 تاریخچه مکالمات")

    if st.session_state.conversation_history:
        # دکمه پاک‌سازی تاریخچه
        if st.button("🗑️ پاک‌سازی تاریخچه", type="secondary", key="clear_history"):
            st.session_state.conversation_history = []
            st.rerun()

        for i, item in enumerate(st.session_state.conversation_history):
            with st.expander(f"❓ {item['query'][:100]}{'...' if len(item['query']) > 100 else ''}", expanded=False):
                col_a1, col_a2 = st.columns([1, 1])
                with col_a1:
                    st.markdown("**پاسخ:**")
                    st.info(item["answer"])
                with col_a2:
                    st.markdown("**اطلاعات:**")
                    st.metric("تعداد چانک‌ها", len(item["chunks"]))
                    st.metric("زمان پاسخ", f"{item['elapsed']:.1f} ثانیه")
                    st.caption(f"ساعت: {item['time']}")
    else:
        st.info("هنوز سؤالی پرسیده نشده است.")


# ============================================================================
# صفحه تنظیمات
# ============================================================================

def render_settings_page():
    """صفحه تنظیمات"""
    st.markdown("# ⚙️ تنظیمات")
    st.markdown("---")

    settings = st.session_state.settings

    # تب‌های تنظیمات
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 مدل LLM", "🔍 بازیابی", "📝 امبدینگ", "💾 ذخیره/بارگذاری"])

    with tab1:
        st.markdown("### تنظیمات مدل زبانی (LLM)")

        with st.form("llm_settings"):
            settings["api_url"] = st.text_input(
                "آدرس API",
                value=settings.get("api_url", "https://api.gapgpt.app/v1"),
                help="آدرس سرور API مدل زبانی"
            )
            settings["api_key"] = st.text_input(
                "کلید API",
                value=settings.get("api_key", ""),
                type="password",
                help="کلید دسترسی به API"
            )
            settings["llm_model"] = st.text_input(
                "مدل LLM",
                value=settings.get("llm_model", "gapgpt-qwen-3.5"),
                help="نام مدل زبانی بزرگ"
            )
            settings["temperature"] = st.slider(
                "دما (Temperature)",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("temperature", 0.2),
                step=0.05,
                help="میزان خلاقیت پاسخ (کم = دقیق‌تر، زیاد = خلاقانه‌تر)"
            )
            settings["max_tokens"] = st.slider(
                "حداکثر توکن",
                min_value=64,
                max_value=4096,
                value=settings.get("max_tokens", 1024),
                step=64,
                help="حداکثر تعداد توکن در پاسخ"
            )
            settings["system_prompt"] = st.text_area(
                "دستور سیستم",
                value=settings.get("system_prompt", "شما یک دستیار فارسی هستید که پاسخ‌ها را به صورت خلاصه اما دقیق ارائه می‌دهد."),
                height=100,
                help="دستورالعمل سیستم برای تنظیم رفتار مدل"
            )

            submitted = st.form_submit_button("💾 ذخیره تنظیمات LLM", type="primary")
            if submitted:
                save_settings()
                # بازنشانی عامل
                st.session_state.rag_agent = None
                st.session_state.llm = None

    with tab2:
        st.markdown("### تنظیمات بازیابی اسناد")

        with st.form("retrieval_settings"):
            settings["top_k"] = st.slider(
                "تعداد چانک‌های بازیابی شده (top_k)",
                min_value=1,
                max_value=20,
                value=settings.get("top_k", 5),
                step=1,
                help="تعداد بخش‌های مرتبط برای هر سؤال"
            )
            st.info(f"در حال حاضر: {settings['top_k']} چانک برای هر سؤال بازیابی می‌شود.")

            submitted = st.form_submit_button("💾 ذخیره تنظیمات بازیابی", type="primary")
            if submitted:
                save_settings()
                st.session_state.rag_agent = None

    with tab3:
        st.markdown("### تنظیمات امبدینگ")

        with st.form("embedding_settings"):
            embedding_models = [
                ("text-embedding-3-small (API)", "api_text-embedding-3-small"),
                ("intfloat/multilingual-e5-small (محلی)", "local_multilingual-e5-small"),
                ("sentence-transformers/all-MiniLM-L6-v2 (محلی)", "local_all-MiniLM-L6-v2"),
            ]

            model_options = {m[1]: m[0] for m in embedding_models}
            current_key = settings.get("embedding_type", "api") + "_" + settings.get("embedding_model", "text-embedding-3-small")

            selected_display = st.selectbox(
                "مدل امبدینگ",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                help="انتخاب مدل برای تولید امبینگ متن"
            )

            settings["embedding_type"] = selected_display.split("_")[0] if "_" in selected_display else "api"
            settings["embedding_model"] = "_".join(selected_display.split("_")[1:]) if "_" in selected_display else "text-embedding-3-small"

            submitted = st.form_submit_button("💾 ذخیره تنظیمات امبدینگ", type="primary")
            if submitted:
                save_settings()

    with tab4:
        st.markdown("### ذخیره و بارگذاری تنظیمات")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 ذخیره تنظیمات", type="primary"):
                save_settings()
        with col2:
            if st.button("📂 بارگذاری تنظیمات", type="secondary"):
                load_settings()

        st.markdown("---")
        st.markdown("### به‌روزرسانی سیستم")
        st.info("پس از تغییر تنظیمات، ممکن است نیاز باشد عامل RAG را از نو راه‌اندازی کنید.")


# ============================================================================
# صفحه آموزشی
# ============================================================================

def render_education_page():
    """صفحه آموزشی - آموزش مفاهیم RAG"""
    st.markdown("# 📖 آموزش مفاهیم RAG")
    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h2>🤔 سیستم RAG چیست؟</h2>
        <p>
            <strong>RAG</strong> مخفف <em>Retrieval-Augmented Generation</em> یعنی <strong>تولید تقویت‌شده با بازیابی</strong> 
            یک فناوری هوش مصنوعی است که ترکیبی از دو سیستم است:
        </p>
        <ul>
            <li><strong>🔍 بازیابی (Retrieval):</strong> پیدا کردن اطلاعات مرتبط از یک پایگاه دانش</li>
            <li><strong>✍️ تولید (Generation):</strong> ساخت پاسخ با استفاده از مدل زبانی بزرگ</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # معماری سیستم
    st.markdown("## 🏗️ معماری سیستم")
    
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │   سؤال کاربر │ ──► │  بازیابی     │ ──► │   امتیازدهی │ ──► │  انتخاب     │
    │   (Query)   │     │  Chunks      │     │  و رتبه‌بندی │     │  مرتبط‌ها   │
    └─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                             │
    ┌─────────────┐     ┌──────────────┐                     │
    │   پاسخ نهایی │ ◄── │   تولید      │ ◄─── ─ ─ ─ ─ ─ ─ ─ ┘
    │  (Response) │     │  پاسخ با LLM │      متن‌های مرتبط
    └─────────────┘     └──────────────┘
    ```
    """)

    # مراحل سیستم
    st.markdown("---")
    st.markdown("## 📋 مراحل سیستم RAG")

    # مرحله 1
    with st.expander("1️⃣ مرحله اول: پردازش و تبدیل سند (PDF Processing)", expanded=False):
        st.markdown("""
        ### تبدیل PDF به متن قابل پردازش
        
        در این مرحله، فایل PDF ورودی پردازش می‌شود:
        
        - **استخراج متن:** متن از فایل PDF با استفاده از PyMuPDF استخراج می‌شود
        - **تمیزسازی:** کاراکترهای اضافی، bullets و شماره‌گذاری‌ها حذف می‌شوند
        - **ادغام خطوط:** خطوط شکسته به پاراگراف‌های منسجم ادغام می‌شوند
        - **ذخیره JSON:** متن تمیز شده در فرمت JSON ذخیره می‌شود
        
        **نکته:** این مرحله فقط یکبار هنگام آپلود سند انجام می‌شود.
        """)

    # مرحله 2
    with st.expander("2️⃣ مرحله دوم: تقسیم به چانک (Chunking)", expanded=False):
        st.markdown("""
        ### تقسیم متن به بخش‌های کوچک‌تر
        
        متن بزرگ به بخش‌های کوچک‌تر تقسیم می‌شود تا پردازش آن آسان‌تر باشد:
        
        - **اندازه چانک:** هر چانک حدود 700 کاراکتر است
        - **همپوشانی (Overlap):** 2 جمله از چانک قبلی برای حفظ ارتباط معنایی
        - **تعداد جملات:** از hazm برای tokenize جملات استفاده می‌شود
        
        **چرا چانک‌بندی؟**
        - مدل‌های زبانی محدودیت طول دارند
        - جستجو در بخش‌های کوچک‌تر دقیق‌تر است
        - هر چانک مستقل قابل پردازش است
        """)

    # مرحله 3
    with st.expander("3️⃣ مرحله سوم: ساخت امبدینگ (Embedding)", expanded=False):
        st.markdown("""
        ### تبدیل متن به بردار عددی
        
        هر چانک به یک بردار عددی (امبدینگ) تبدیل می‌شود:
        
        - **معنی به عدد:** هر متن به یک بردار چندبعدی تبدیل می‌شود
        - **شباهت معنایی:** متن‌های مشابه → بردارهای مشابه
        - **مدل‌های پشتیبانی:** 
          - `text-embedding-3-small` (API)
          - `intfloat/multilingual-e5-small` (محلی)
        
        **مزیت:** جستجوی معنایی به جای جستجوی کلمه‌ای
        """)

    # مرحله 4
    with st.expander("4️⃣ مرحله چهارم: ذخیره در FAISS (Indexing)", expanded=False):
        st.markdown("""
        ### ایجاد شاخص جستجوی سریع
        
        امبینگ‌ها در FAISS ذخیره می‌شوند:
        
        - **FAISS:** کتابخانه جستجوی نزدیک‌ورد (Near Neighbor Search) از متا
        - **IndexFlatIP:** استفاده از ضرب داخلی برای شباهت کسینوسی
        - **سرعت بالا:** جستجوی میلی‌ثانیه‌ای در میان میلیون‌ها چانک
        - **نرمال‌سازی:** بردارها برای مقایسه دقیق‌تر نرمال می‌شوند
        """)

    # مرحله 5
    with st.expander("5️⃣ مرحله پنجم: بازیابی (Retrieval)", expanded=False):
        st.markdown("""
        ### پیدا کردن چانک‌های مرتبط با سؤال
        
        وقتی سؤال پرسیده می‌شود:
        
        1. **امبدینگ سؤال:** سؤال به بردار عددی تبدیل می‌شود
        2. **جستجوی FAISS:** نزدیک‌ترین بردارها پیدا می‌شوند
        3. **فیلترینگ:** نتایج بر اساس آستانه امتیاز فیلتر می‌شوند
        4. **رتبه‌بندی:** نتایج بر اساس شباهت مرتب می‌شوند
        5. **انتخاب top_k:** بهترین n چانک انتخاب می‌شوند
        
        **پارامترهای مهم:**
        - `top_k`: تعداد نتایج
        - `abs_min_score`: آستانه حداقل امتیاز
        - `rel_score_drop`: آستانه نسبی
        """)

    # مرحله 6
    with st.expander("6️⃣ مرحله ششم: تولید پاسخ (Generation)", expanded=False):
        st.markdown("""
        ### ساخت پاسخ نهایی با LLM
        
        در این مرحله، مدل زبانی بزرگ پاسخ را می‌سازد:
        
        - **Context:** چانک‌های بازیابی شده به عنوان زمینه
        - **Question:** سؤال کاربر
        - **System Prompt:** دستورالعمل سیستم
        - **Temperature:** کنترل خلاقیت پاسخ
        - **Max Tokens:** حداکثر طول پاسخ
        
        **فرآیند:**
        1. سؤال + متن‌های مرتبط → Prompt
        2. ارسال به API مدل زبانی
        3. دریافت و نمایش پاسخ
        """)

    # تفاوت جستجوی معمولی و RAG
    st.markdown("---")
    st.markdown("## 🔍 مقایسه جستجوی معمولی vs RAG")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❌ جستجوی معمولی (Keyword Search)")
        st.markdown("""
        - جستجوی کلمه کلیدی
        - نیاز به تطبیق دقیق کلمات
        - عدم درک معنی
        - نتایج ناقص
        """)
        st.code(
            "جستجو: 'ثبت نام'\n"
            "نتیجه: فقط صفحات دارای 'ثبت' و 'نام'",
            language="text"
        )

    with col2:
        st.markdown("### ✅ RAG (معنایی)")
        st.markdown("""
        - درک معنای سؤال
        - پیدا کردن مفاهیم مرتبط
        - پاسخ طبیعی و روان
        - منبع‌دهی دقیق
        """)
        st.code(
            "جستجو: 'چطور عضو خانواده اضافه کنم؟'\n"
            "نتیجه: پیدا کردن بخش 'ثبت‌نام عضو جدید'",
            language="text"
        )

    # نکات کلیدی
    st.markdown("---")
    st.markdown("## 💡 نکات کلیدی برای استفاده بهتر")

    st.markdown("""
    1. **سؤالات دقیق‌تر:** سؤالات مشخص و دقیق‌تر → پاسخ‌های بهتر
    2. **تنظیم top_k:** افزایش top_k → زمینه بیشتر، احتمال خطا بیشتر
    3. **Temperature پایین:** برای پاسخ‌های دقیق‌تر، دما را کم کنید
    4. **کیفیت سند:** اسناد تمیز و ساختاریافته → نتایج بهتر
    """)


# ============================================================================
# برنامه اصلی Streamlit
# ============================================================================

def main():
    """تابع اصلی برنامه"""
    # راه‌اندازی session state
    init_session_state()

    # نوار کناری - ناوبری
    with st.sidebar:
        st.markdown("## 📚 سیستم RAG فارسی")
        st.markdown("---")

        # منوی ناوبری
        nav_items = [
            ("🏠 خانه", "home"),
            ("📤 آپلود اسناد", "upload"),
            ("💬 پرسش و پاسخ", "qa"),
            ("⚙️ تنظیمات", "settings"),
            ("📖 آموزش", "education"),
        ]

        for label, page_id in nav_items:
            if st.button(
                label,
                key=f"nav_{page_id}",
                type="primary" if st.session_state.current_page == page_id else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")

        # اطلاعات وضعیت
        st.markdown("### 📊 وضعیت سیستم")
        
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            st.success(f"✅ {len(chunks)} چانک")
        else:
            st.warning("⚠️ بدون چانک")

        if EMBEDDINGS_PATH.exists():
            st.success("✅ امبدینگ‌ها آماده")
        else:
            st.warning("⚠️ بدون امبدینگ")

        agent_ready = st.session_state.rag_agent is not None
        if agent_ready:
            st.success("✅ عامل RAG فعال")
        else:
            st.info("ℹ️ عامل RAG غیرفعال")

        st.markdown("---")
        
        # دکمه تازه‌سازی
        if st.button("🔄 تازه‌سازی", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.rag_agent = None
            st.session_state.retriever = None
            st.session_state.llm = None
            st.rerun()

    # نمایش صفحه فعال
    if st.session_state.current_page == "home":
        render_home_page()
    elif st.session_state.current_page == "upload":
        render_upload_page()
    elif st.session_state.current_page == "qa":
        render_qa_page()
    elif st.session_state.current_page == "settings":
        render_settings_page()
    elif st.session_state.current_page == "education":
        render_education_page()


if __name__ == "__main__":
    main()