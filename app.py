import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
import json
import os
from typing import Dict, Optional
from PIL import Image
import base64
from io import BytesIO
import time

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="信用卡效益最大化助手",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "credit_cards.db"

BANKS = [
    "招商银行", "工商银行", "建设银行", "中国银行", "农业银行",
    "交通银行", "中信银行", "光大银行", "民生银行", "浦发银行",
    "兴业银行", "平安银行", "广发银行", "华夏银行", "邮储银行",
    "北京银行", "上海银行", "江苏银行", "宁波银行", "南京银行",
    "杭州银行", "其他"
]

CATEGORIES = [
    "餐饮", "超市/日用", "网购", "加油", "出行/交通", "酒店旅游",
    "医疗", "教育", "娱乐", "保险", "税费", "其他", "全场景"
]

# ==================== 数据库 ====================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank TEXT NOT NULL,
            card_name TEXT NOT NULL,
            last4 TEXT,
            annual_fee REAL DEFAULT 0,
            fee_waiver TEXT,
            main_categories TEXT,
            points_value REAL DEFAULT 0.01,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank TEXT,
            card_name TEXT,
            title TEXT NOT NULL,
            category TEXT,
            rebate_type TEXT,
            rebate_value REAL,
            max_rebate REAL,
            min_spend REAL DEFAULT 0,
            start_date TEXT,
            end_date TEXT,
            conditions TEXT,
            source TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY,
            monthly_budget TEXT,
            preferred_redemption TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def df_from_query(query: str, params=()):
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def execute_sql(sql: str, params=()):
    conn = get_conn()
    c = conn.cursor()
    c.execute(sql, params)
    conn.commit()
    last_id = c.lastrowid
    conn.close()
    return last_id

def get_cards_df():
    return df_from_query("SELECT * FROM cards ORDER BY bank, card_name")

def get_promos_df(active_only=True):
    sql = "SELECT * FROM promotions"
    if active_only:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY end_date DESC, bank"
    return df_from_query(sql)

# ==================== 大模型配置 ====================
def get_llm_config():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o")
    return api_key, base_url, model

# ==================== 大模型解析 ====================
def parse_promo_with_llm(text: str, api_key: str, base_url: str, model: str) -> Optional[Dict]:
    if not api_key or not text.strip():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""
你是一个信用卡优惠信息提取专家。请从以下中文活动文本中提取结构化信息，严格只返回 JSON：
{{
  "bank": "银行名称",
  "card_name": "适用卡种，没有则空字符串",
  "title": "活动标题（简短）",
  "category": "最匹配的消费类别，从 {CATEGORIES} 中选一个，或全场景",
  "rebate_type": "返现 / 积分倍率 / 立减 / 其他",
  "rebate_value": 数字（比例用小数如0.05，倍率用5，立减用金额）,
  "max_rebate": 数字或null,
  "min_spend": 数字或0,
  "start_date": "YYYY-MM-DD 或空字符串",
  "end_date": "YYYY-MM-DD 或空字符串",
  "conditions": "关键条件摘要"
}}
活动文本：
{text[:3500]}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        st.error(f"文字解析失败: {e}")
        return None

def parse_promo_from_image(image: Image.Image, api_key: str, base_url: str, model: str) -> Optional[Dict]:
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        buffered = BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""
你是一个专业的信用卡优惠信息提取助手。请仔细查看这张手机银行App截图，只提取优惠活动相关信息，忽略余额、卡号、个人信息。
严格只返回纯 JSON：
{{
  "bank": "银行名称",
  "card_name": "适用卡种（没有就空字符串）",
  "title": "活动标题（简洁）",
  "category": "最匹配类别，从 {CATEGORIES} 选一个，或全场景",
  "rebate_type": "返现 / 积分倍率 / 立减 / 其他",
  "rebate_value": 数字（5%写成0.05，5倍写成5，立减50写成50）,
  "max_rebate": 数字或null,
  "min_spend": 数字或0,
  "start_date": "YYYY-MM-DD 或空字符串",
  "end_date": "YYYY-MM-DD 或空字符串",
  "conditions": "关键条件、上限、参与方式摘要"
}}
如果有多个活动，只提取最主要的一个。
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": "high"}}
                ]
            }],
            temperature=0.1,
            max_tokens=1200
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        st.error(f"图片解析失败: {str(e)}")
        return None

# ==================== 推荐引擎 ====================
def recommend_detailed(consumption: Dict[str, float], cards_df: pd.DataFrame, promos_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for cat, amount in consumption.items():
        if amount <= 0:
            continue
        best_score = -999999
        best_row = None
        
        cat_promos = promos_df[
            (promos_df["category"].astype(str).str.contains(cat, na=False)) |
            (promos_df["category"] == "全场景")
        ]
        
        for _, promo in cat_promos.iterrows():
            matched_cards = cards_df
            if promo["bank"]:
                matched_cards = matched_cards[matched_cards["bank"] == promo["bank"]]
            if promo["card_name"] and str(promo["card_name"]).strip():
                matched_cards = matched_cards[matched_cards["card_name"].astype(str).str.contains(str(promo["card_name"]), na=False)]
            
            if matched_cards.empty and promo["bank"]:
                matched_cards = cards_df[cards_df["bank"] == promo["bank"]]
            if matched_cards.empty:
                matched_cards = cards_df
            
            for _, card in matched_cards.iterrows():
                rebate = 0.0
                rtype = str(promo["rebate_type"] or "")
                rval = float(promo["rebate_value"] or 0)
                maxr = promo["max_rebate"]
                
                if "返现" in rtype:
                    rebate = amount * rval
                    if maxr: rebate = min(rebate, float(maxr))
                elif "积分" in rtype or "倍率" in rtype:
                    points = amount * max(rval, 1)
                    rebate = points * float(card["points_value"] or 0.01)
                elif "立减" in rtype:
                    rebate = rval
                
                monthly_fee = float(card["annual_fee"] or 0) / 12
                net = rebate - monthly_fee * 0.25
                
                if net > best_score:
                    best_score = net
                    best_row = {
                        "消费类别": cat,
                        "计划金额": amount,
                        "推荐银行": card["bank"],
                        "推荐卡": card["card_name"],
                        "卡后四位": card["last4"] or "",
                        "匹配活动": promo["title"],
                        "预估收益(元)": round(rebate, 2),
                        "净收益粗估": round(net, 2),
                        "活动条件": promo["conditions"] or "",
                        "有效期": f"{promo['start_date'] or ''} ~ {promo['end_date'] or ''}",
                        "备注": card["notes"] or ""
                    }
        
        if best_row is None:
            for _, card in cards_df.iterrows():
                mains = []
                try:
                    mains = json.loads(card["main_categories"] or "[]")
                except:
                    pass
                if cat in mains or "全场景" in mains or not mains:
                    best_row = {
                        "消费类别": cat,
                        "计划金额": amount,
                        "推荐银行": card["bank"],
                        "推荐卡": card["card_name"],
                        "卡后四位": card["last4"] or "",
                        "匹配活动": "卡片基础权益",
                        "预估收益(元)": 0,
                        "净收益粗估": 0,
                        "活动条件": "请查看卡片自身权益",
                        "有效期": "-",
                        "备注": card["notes"] or ""
                    }
                    break
        
        if best_row:
            results.append(best_row)
    
    return pd.DataFrame(results)

def recommend_flexible(cards_df: pd.DataFrame, promos_df: pd.DataFrame) -> Dict:
    active = promos_df[promos_df["is_active"] == 1].copy() if not promos_df.empty else pd.DataFrame()
    
    high_value = pd.DataFrame()
    if not active.empty:
        high_value = active.sort_values("rebate_value", ascending=False).head(12)
    
    bank_strength = []
    for bank in cards_df["bank"].unique():
        bank_cards = cards_df[cards_df["bank"] == bank]
        bank_promos = active[active["bank"] == bank] if not active.empty else pd.DataFrame()
        score = len(bank_promos) * 3 + len(bank_cards) * 0.5 - (bank_cards["annual_fee"].mean() or 0) * 0.01
        bank_strength.append({
            "银行": bank,
            "卡片数": len(bank_cards),
            "活跃活动数": len(bank_promos),
            "推荐指数": round(score, 2)
        })
    
    strength_df = pd.DataFrame(bank_strength).sort_values("推荐指数", ascending=False)
    
    suggestions = [
        "1. 高频小额消费（餐饮、超市、网购）固定用1-2张高返现/高倍率卡。",
        "2. 大额或特定场景单独匹配当前最优活动。",
        "3. 每月初更新活动（用截图解析最快）。",
        "4. 年费卡必须计算权益能否覆盖年费。",
        "5. 详细模式适合有明确消费计划时使用。"
    ]
    
    return {
        "高价值活动Top": high_value[["bank", "title", "category", "rebate_type", "rebate_value", "max_rebate", "end_date"]] if not high_value.empty else pd.DataFrame(),
        "银行推荐指数": strength_df,
        "弹性策略建议": suggestions
    }

# ==================== 页面函数 ====================
def page_dashboard():
    st.header("📊 仪表盘")
    cards = get_cards_df()
    promos = get_promos_df()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("卡片总数", len(cards))
    c2.metric("活跃优惠", len(promos))
    c3.metric("覆盖银行", cards["bank"].nunique() if not cards.empty else 0)
    c4.metric("数据存储", "本地临时")
    
    if not cards.empty:
        st.subheader("卡片分布")
        st.bar_chart(cards["bank"].value_counts())
    
    st.subheader("即将到期活动（30天内）")
    if not promos.empty:
        try:
            promos = promos.copy()
            promos["end_date_dt"] = pd.to_datetime(promos["end_date"], errors="coerce")
            soon = promos[promos["end_date_dt"] <= (datetime.now() + timedelta(days=30))]
            st.dataframe(soon[["bank", "title", "category", "end_date", "rebate_value"]].head(15), use_container_width=True)
        except:
            st.dataframe(promos.head(10), use_container_width=True)
    else:
        st.info("暂无活动数据")

def page_cards():
    st.header("💳 我的卡片")
    
    with st.expander("➕ 添加新卡片", expanded=False):
        with st.form("card_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            bank = col1.selectbox("银行", BANKS)
            card_name = col2.text_input("卡名")
            last4 = col1.text_input("卡号后四位")
            annual_fee = col2.number_input("年费（元）", min_value=0.0, value=0.0, step=50.0)
            fee_waiver = st.text_input("年费减免条件")
            main_cats = st.multiselect("主要权益类别", CATEGORIES)
            points_value = st.number_input("1积分约等于多少元", min_value=0.0, value=0.01, step=0.005, format="%.3f")
            notes = st.text_area("备注")
            
            if st.form_submit_button("保存卡片", type="primary"):
                if card_name.strip():
                    now = datetime.now().isoformat()
                    execute_sql(
                        """INSERT INTO cards (bank, card_name, last4, annual_fee, fee_waiver, main_categories, points_value, notes, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (bank, card_name.strip(), last4, annual_fee, fee_waiver, 
                         json.dumps(main_cats, ensure_ascii=False), points_value, notes, now, now)
                    )
                    st.success("添加成功！")
                    st.rerun()
    
    cards = get_cards_df()
    if not cards.empty:
        st.dataframe(cards[["id", "bank", "card_name", "last4", "annual_fee", "points_value", "notes"]], use_container_width=True)
        del_id = st.number_input("删除卡片 ID", min_value=1, step=1)
        if st.button("删除卡片", type="secondary"):
            execute_sql("DELETE FROM cards WHERE id = ?", (int(del_id),))
            st.success("已删除")
            st.rerun()
    else:
        st.info("还没有卡片，请先添加")

def page_promotions():
    st.header("🎯 优惠活动管理")
    st.warning("【隐私】上传截图前请裁剪，只保留活动区域，打码余额和完整卡号！")
    
    api_key, base_url, model = get_llm_config()
    
    with st.sidebar.expander("🔑 API 配置", expanded=not bool(api_key)):
        if not api_key:
            api_key = st.text_input("API Key", type="password")
            base_url = st.text_input("Base URL", value=base_url)
            model = st.text_input("模型（需支持视觉）", value=model)
        else:
            st.success("已从 Secrets 读取配置")
            st.caption(f"模型: {model}")
    
    tab1, tab2 = st.tabs(["截图/文字添加", "活动列表"])
    
    with tab1:
        st.subheader("截图上传解析")
        uploaded_files = st.file_uploader("上传手机银行截图", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        
        if uploaded_files:
            for idx, f in enumerate(uploaded_files):
                image = Image.open(f)
                st.image(image, caption=f"图片 {idx+1}", use_container_width=True)
                if st.button(f"解析图片 {idx+1}", key=f"p{idx}"):
                    if not api_key:
                        st.error("请配置 API Key")
                    else:
                        with st.spinner("识别中..."):
                            data = parse_promo_from_image(image, api_key, base_url, model)
                            if data:
                                st.session_state["parsed_promo"] = data
                                st.success("解析成功！请检查下方表单")
                                st.json(data)
        
        st.markdown("---")
        raw_text = st.text_area("或粘贴活动文字")
        if st.button("文字解析"):
            if api_key and raw_text.strip():
                data = parse_promo_with_llm(raw_text, api_key, base_url, model)
                if data:
                    st.session_state["parsed_promo"] = data
                    st.json(data)
        
        parsed = st.session_state.get("parsed_promo", {})
        with st.form("promo_form"):
            col1, col2 = st.columns(2)
            bank_idx = BANKS.index(parsed.get("bank")) if parsed.get("bank") in BANKS else len(BANKS)-1
            bank = col1.selectbox("银行", BANKS, index=bank_idx)
            card_name = col2.text_input("适用卡种", value=str(parsed.get("card_name") or ""))
            title = st.text_input("活动标题 *", value=str(parsed.get("title") or ""))
            
            cat_val = parsed.get("category") if parsed.get("category") in CATEGORIES else "其他"
            category = col1.selectbox("类别", CATEGORIES, index=CATEGORIES.index(cat_val))
            
            rtype_options = ["返现", "积分倍率", "立减", "其他"]
            rtype_val = parsed.get("rebate_type") if parsed.get("rebate_type") in rtype_options else "返现"
            rebate_type = col2.selectbox("优惠类型", rtype_options, index=rtype_options.index(rtype_val))
            
            rebate_value = col1.number_input("优惠值", value=float(parsed.get("rebate_value") or 0.0), format="%.4f")
            max_rebate = col2.number_input("上限", value=float(parsed.get("max_rebate") or 0.0))
            min_spend = col1.number_input("最低消费", value=float(parsed.get("min_spend") or 0.0))
            
            try:
                s_date = date.fromisoformat(parsed["start_date"]) if parsed.get("start_date") else date.today()
            except:
                s_date = date.today()
            try:
                e_date = date.fromisoformat(parsed["end_date"]) if parsed.get("end_date") else date.today() + timedelta(days=30)
            except:
                e_date = date.today() + timedelta(days=30)
            
            start_date = col2.date_input("开始日期", value=s_date)
            end_date = col1.date_input("结束日期", value=e_date)
            conditions = st.text_area("条件说明", value=str(parsed.get("conditions") or ""))
            
            if st.form_submit_button("💾 保存活动", type="primary"):
                if title.strip():
                    now = datetime.now().isoformat()
                    execute_sql(
                        """INSERT INTO promotions 
                           (bank, card_name, title, category, rebate_type, rebate_value, max_rebate, min_spend,
                            start_date, end_date, conditions, source, is_active, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                        (bank, card_name, title.strip(), category, rebate_type, rebate_value, max_rebate, min_spend,
                         str(start_date), str(end_date), conditions, "截图/解析", now, now)
                    )
                    st.success("保存成功！")
                    if "parsed_promo" in st.session_state:
                        del st.session_state["parsed_promo"]
                    time.sleep(0.5)
                    st.rerun()
    
    with tab2:
        promos = get_promos_df(active_only=False)
        if not promos.empty:
            st.dataframe(promos, use_container_width=True)
            del_pid = st.number_input("删除活动 ID", min_value=1, step=1)
            if st.button("删除活动"):
                execute_sql("DELETE FROM promotions WHERE id = ?", (int(del_pid),))
                st.success("已删除")
                st.rerun()
        else:
            st.info("暂无活动")

def page_recommend():
    st.header("🧠 智能推荐")
    cards = get_cards_df()
    promos = get_promos_df()
    
    if cards.empty:
        st.warning("请先添加卡片")
        return
    
    mode = st.radio("推荐模式", ["详细模式", "弹性模式"], horizontal=True)
    
    if mode == "详细模式":
        st.subheader("输入消费规划（元）")
        consumption = {}
        cols = st.columns(4)
        for i, cat in enumerate(CATEGORIES):
            with cols[i % 4]:
                val = st.number_input(cat, min_value=0.0, value=0.0, step=100.0, key=f"c_{cat}")
                if val > 0:
                    consumption[cat] = val
        
        if st.button("生成推荐", type="primary", use_container_width=True):
            if consumption:
                result_df = recommend_detailed(consumption, cards, promos)
                if not result_df.empty:
                    st.dataframe(result_df, use_container_width=True)
                    st.metric("预估总收益", f"¥ {result_df['预估收益(元)'].sum():.2f}")
                    csv = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("下载CSV", csv, "recommend.csv", "text/csv")
                else:
                    st.info("暂无匹配结果")
    else:
        flex = recommend_flexible(cards, promos)
        st.write("#### 高价值活动")
        if not flex["高价值活动Top"].empty:
            st.dataframe(flex["高价值活动Top"], use_container_width=True)
        st.write("#### 银行推荐指数")
        st.dataframe(flex["银行推荐指数"], use_container_width=True)
        st.write("#### 策略建议")
        for s in flex["弹性策略建议"]:
            st.markdown(f"- {s}")

def page_import():
    st.header("📥 数据导入（从 HTML 版或备份恢复）")
    st.info("上传之前从 HTML 版导出的 JSON 文件，将自动合并卡片和活动数据（去重）。")
    
    uploaded_file = st.file_uploader("选择 JSON 备份文件", type=["json"])
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.json(data)  # 预览
            if st.button("确认导入并合并", type="primary"):
                # 导入卡片
                if "cards" in data:
                    for card in data["cards"]:
                        exists = df_from_query("SELECT id FROM cards WHERE bank=? AND card_name=?", 
                                             (card.get("bank", ""), card.get("name", "")))
                        if exists.empty:
                            now = datetime.now().isoformat()
                            execute_sql(
                                """INSERT INTO cards (bank, card_name, last4, annual_fee, fee_waiver, main_categories, points_value, notes, created_at, updated_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (card.get("bank", ""), card.get("name", ""), "", 
                                 card.get("feeAnnual", 0), "", json.dumps([], ensure_ascii=False),
                                 0.01, card.get("benefits", ""), now, now)
                            )
                
                # 导入活动（简化映射）
                if "promos" in data:
                    count = 0
                    for promo in data["promos"]:
                        exists = df_from_query("SELECT id FROM promotions WHERE title=? AND bank=?", 
                                             (promo.get("name", ""), promo.get("bank", "")))
                        if exists.empty:
                            now = datetime.now().isoformat()
                            type_map = {"discount": "立减", "fullReduction": "立减", "cashback": "返现", "points": "积分倍率", "gift": "其他"}
                            execute_sql(
                                """INSERT INTO promotions (bank, card_name, title, category, rebate_type, rebate_value, max_rebate, min_spend, start_date, end_date, conditions, source, is_active, created_at, updated_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                                (promo.get("bank", ""), "", promo.get("name", ""), 
                                 promo.get("category", "其他"), type_map.get(promo.get("type", ""), "其他"),
                                 promo.get("value", 0), promo.get("max", 0), 0,
                                 date.today().isoformat(), promo.get("expiry", ""), 
                                 promo.get("merchant", ""), "JSON导入", now, now)
                            )
                            count += 1
                    st.success(f"导入完成！新增 {count} 个活动")
                else:
                    st.success("卡片/数据导入完成！")
                st.rerun()
        except Exception as e:
            st.error(f"导入失败: {e}")

def page_analysis():
    st.header("📈 消费账单分析")
    st.info("上传支付宝/微信/银行账单 CSV 文件，自动分析消费结构并匹配最优卡片建议。")
    
    uploaded_files = st.file_uploader("上传账单 CSV（支持多文件）", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        all_data = []
        for f in uploaded_files:
            try:
                df = pd.read_csv(f, encoding="utf-8")
                all_data.append(df)
                st.subheader(f"📄 {f.name}")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 简单分析示例
                if "金额" in df.columns or "price" in df.columns or "amount" in df.columns:
                    col_name = [c for c in df.columns if c in ["金额", "price", "amount", "消费金额"]][0]
                    total = df[col_name].sum()
                    st.metric("总消费", f"¥{total:,.2f}")
            except Exception as e:
                st.error(f"读取 {f.name} 失败: {e}")
        
        if all_data:
            st.success(f"成功加载 {len(all_data)} 个文件")
            st.info("💡 提示：后续可扩展自动按商户匹配最优信用卡、生成月度消费报告等功能。")

def page_notifications():
    st.header("🔔 活动到期提醒")
    promos = get_promos_df()
    
    if promos.empty:
        st.info("暂没有活动数据")
        return
    
    # 转换为日期
    promos = promos.copy()
    promos["end_date_dt"] = pd.to_datetime(promos["end_date"], errors="coerce")
    
    # 最近 3 天
    soon = promos[promos["end_date_dt"] <= (datetime.now() + timedelta(days=3))]
    # 本周到期
    week = promos[(promos["end_date_dt"] <= (datetime.now() + timedelta(days=7))) & (promos["end_date_dt"] > (datetime.now() + timedelta(days=3)))]
    
    if not soon.empty:
        st.error(f"⚠️ **紧急：{len(soon)} 个活动将在 3 天内到期！**")
        st.dataframe(soon[["bank", "title", "end_date", "conditions"]], use_container_width=True)
    
    if not week.empty:
        st.warning(f"📅 本周还有 {len(week)} 个活动即将到期")
        with st.expander("查看本周到期活动"):
            st.dataframe(week[["bank", "title", "end_date", "conditions"]], use_container_width=True)
    
    if soon.empty and week.empty:
        st.success("✅ 近期没有即将到期的活动")

def page_export():
    st.header("📤 导出备份")
    st.info("Streamlit Cloud 免费版重启后数据会丢失，请定期导出备份！")
    
    cards = get_cards_df()
    promos = get_promos_df(active_only=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("卡片数据")
        if not cards.empty:
            export_cards = cards.copy()
            if "main_categories" in export_cards.columns:
                export_cards["main_categories"] = export_cards["main_categories"].apply(
                    lambda x: ", ".join(json.loads(x)) if x else ""
                )
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                export_cards.to_excel(writer, index=False, sheet_name="cards")
            st.download_button("下载卡片 Excel", buffer.getvalue(),
                               f"cards_{datetime.now().strftime('%Y%m%d')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("暂无卡片")
    
    with col2:
        st.subheader("活动数据")
        if not promos.empty:
            buffer2 = BytesIO()
            with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
                promos.to_excel(writer, index=False, sheet_name="promotions")
            st.download_button("下载活动 Excel", buffer2.getvalue(),
                               f"promos_{datetime.now().strftime('%Y%m%d')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("暂无活动")
    
    st.markdown("---")
    st.write(f"本地数据库路径：`{os.path.abspath(DB_PATH)}`")

def page_help():
    st.header("使用说明与部署")
    st.markdown("""
### 功能说明
- **截图解析**：上传手机银行截图，自动提取活动信息
- **智能推荐**：详细模式（精确匹配）和弹性模式（策略建议）
- **数据互通**：支持从 HTML 版导入 JSON 数据
- **账单分析**：上传 CSV 账单文件进行分析
- **活动提醒**：查看即将到期活动

### 部署
已部署到 Streamlit Cloud，数据存储在临时空间，请定期导出备份。

### 隐私
- 截图请脱敏后再上传
- API Key 存储在 Streamlit Secrets，不会公开
    """)

# ==================== 主程序 ====================
def main():
    st.sidebar.title("💳 CreditTool")
    st.sidebar.caption("信用卡效益最大化助手")
    
    pages = {
        "📊 仪表盘": page_dashboard,
        "💳 我的卡片": page_cards,
        "🎯 优惠活动": page_promotions,
        "🧠 智能推荐": page_recommend,
        "📥 数据导入": page_import,
        "📈 账单分析": page_analysis,
        "🔔 活动提醒": page_notifications,
        "📤 导出备份": page_export,
        "📖 使用说明": page_help,
    }
    
    choice = st.sidebar.radio("导航", list(pages.keys()))
    pages[choice]()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("数据临时存储，请常备份")

if __name__ == "__main__":
    main()