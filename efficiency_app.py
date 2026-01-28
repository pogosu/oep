import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import newton

# --- CONFIGURATION ---
st.set_page_config(page_title="Project Efficiency Pro", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .metric-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0d6efd;
    }
    .analysis-box {
        border-left: 5px solid #0d6efd;
        padding-left: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

VARIANTS = {
    1: [40, 80, 80, 55],
    2: [50, 80, 80, 55],
    3: [60, 80, 80, 55],
    4: [70, 80, 80, 55],
    5: [70, 80, 70, 55],
    6: [70, 80, 60, 55],
    7: [70, 80, 55, 55],
    8: [70, 80, 55, 60],
    9: [80, 80, 55, 60],
    10: [80, 70, 55, 60],
    11: [80, 60, 55, 60]
}

# --- LOGIC ---
def get_npv(rate, years, net_flows):
    return sum(f / (1 + rate)**t for t, f in zip(years, net_flows))

def calculate_metrics(df, discount_rate):
    years = df['Year'].values
    incomes = df['Inc'].values
    expenses = df['Inv'].values
    net_flows = incomes - expenses
    
    dcf_list = []
    pv_inflows = 0
    pv_outflows = 0
    
    is_distributed = (np.count_nonzero(expenses) > 1)
    
    for t, inc, exp, net in zip(years, incomes, expenses, net_flows):
        # Income Discounting
        dfactor_inc = 1 / (1 + discount_rate)**t
        pv_inc = inc * dfactor_inc
        pv_inflows += pv_inc
        
        # Expense Discounting (k-1 if distributed)
        if is_distributed:
             exp_denom_pow = t - 1
             dfactor_exp = 1 / (1 + discount_rate)**(exp_denom_pow)
        else:
             dfactor_exp = 1.0 
        
        pv_exp = exp * dfactor_exp
        pv_outflows += pv_exp
        
        # DCF for Series (Net)
        dcf = pv_inc - pv_exp 
        dcf_list.append(dcf)

    npv = pv_inflows - pv_outflows
    pi = pv_inflows / pv_outflows if pv_outflows != 0 else float('inf')
    
    # PP
    cum_flow = np.cumsum(net_flows)
    pp, pp_idx = None, -1
    for i in range(len(cum_flow)):
        curr = cum_flow[i]
        prev = cum_flow[i-1] if i > 0 else 0
        if curr >= 0:
            if i == 0:
                pp = 0
            else:
                uncovered = abs(prev)
                nxt = net_flows[i]
                pp = years[i-1] + uncovered / nxt
                pp_idx = i
            break
            
    # DPP
    cum_dcf = np.cumsum(dcf_list)
    dpp, dpp_idx = None, -1
    for i in range(len(cum_dcf)):
        curr = cum_dcf[i]
        prev = cum_dcf[i-1] if i > 0 else 0
        if curr >= 0:
            if i == 0:
                dpp = 0
            else:
                uncovered = abs(prev)
                nxt = dcf_list[i]
                yr_prev = years[i-1]
                dpp = yr_prev + uncovered / nxt
                dpp_idx = i
            break

    # IRR
    def solver(r):
        val = 0
        for t, inc, exp in zip(years, incomes, expenses):
             p_inc = inc / (1+r)**t
             p_exp = exp / (1+r)**(t-1) if is_distributed else exp
             val += p_inc - p_exp
        return val

    try:
        irr = newton(solver, discount_rate) * 100
    except:
        irr = None

    return {
        'NPV': npv, 'PI': pi, 'PP': pp, 'DPP': dpp, 'IRR': irr,
        'PV_In': pv_inflows, 'PV_Out': pv_outflows,
        'PP_Idx': pp_idx, 'DPP_Idx': dpp_idx,
        'Cum_Flow': cum_flow, 'Cum_DCF': cum_dcf,
        'Net_Flows': net_flows, 'DCF_Series': dcf_list,
        'Is_Dist': is_distributed
    }

# --- GUI ---
st.title("💸 Профессиональный анализ проекта")

with st.sidebar:
    st.header("Настройки")
    mode = st.radio("Режим данных", ["По вариантам", "Ручной ввод"], horizontal=True)
    if mode == "По вариантам":
        var_id = st.selectbox("Номер варианта", list(VARIANTS.keys()))
        years_p = 4
        ic_0 = 178.0
    else:
        years_p = st.number_input("Период (лет)", 1, 20, 4)
        ic_0 = st.number_input("Инвестиции (год 1)", value=178.0)
        
    rate_pc = st.number_input("Ставка (%)", value=15.0, step=0.1)
    rate = rate_pc / 100.0
    prec = st.slider("Точность", 0, 5, 2)
    fmt = f"%.{prec}f"

rows = []
if mode == "По вариантам":
    for i, val in enumerate(VARIANTS[var_id]):
        yr = i + 1
        inv = ic_0 if yr == 1 else 0.0
        rows.append({"Year": yr, "Inv": inv, "Inc": float(val)})
else:
    for i in range(1, years_p+1):
        inv = ic_0 if i == 1 else 0.0
        rows.append({"Year": i, "Inv": inv, "Inc": 60.0})

df_input = pd.DataFrame(rows)
st.subheader("1. Исходные данные")
ed_df = st.data_editor(df_input, width='stretch', hide_index=True)
ed_df['NetFlow'] = ed_df['Inc'] - ed_df['Inv']

m2 = calculate_metrics(ed_df, rate)

# Manual IRR Logic
def calc_npv_manual_viz(r_dec, df, is_dist):
    val = 0
    for i, row in df.iterrows():
        t = row['Year']
        inc = row['Inc']
        exp = row['Inv']
        term_inc = inc / (1+r_dec)**t
        term_exp = exp / (1+r_dec)**(t-1) if is_dist else exp
        val += term_inc - term_exp
    return val

st.divider()
st.subheader("2. Результаты и Аналитика")

def conclusion_box(title, value_str, analysis_html, calc_elements, status="info"):
    with st.container(border=True):
        c1, c2 = st.columns([2, 2.5])
        with c1:
            st.markdown(f'<div class="metric-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{value_str}</div>', unsafe_allow_html=True)
            if status == "success": st.success("✅ ПРОЕКТ ЭФФЕКТИВЕН")
            elif status == "error": st.error("❌ ПРОЕКТ НЕЭФФЕКТИВЕН")
            
            st.markdown(f'<div class="analysis-box">{analysis_html}</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown("**Подробный расчет:**")
            for elem in calc_elements:
                if elem['type'] == 'latex':
                    st.latex(elem['content'])
                elif elem['type'] == 'code':
                    st.code(elem['content'], language=None)
                elif elem['type'] == 'html':
                    st.markdown(elem['content'], unsafe_allow_html=True)

# 1. NPV
if m2['Is_Dist']:
    npv_f = r"NPV = \sum \frac{P_k}{(1+r)^k} - \sum \frac{IC_k}{(1+r)^{k-1}}"
    inc_part = " + ".join([f"\\frac{{{row['Inc']}}}{{(1+{rate})^{{{row['Year']}}}}}" for _, row in ed_df.iterrows() if row['Inc']>0])
    exp_part = " + ".join([f"\\frac{{{row['Inv']}}}{{(1+{rate})^{{{row['Year']-1}}}}}" for _, row in ed_df.iterrows() if row['Inv']>0])
    npv_sub = f"NPV = ({inc_part}) - ({exp_part})"
else:
    npv_f = r"NPV = \sum \frac{P_k}{(1+r)^k} - IC"
    inc_part = " + ".join([f"\\frac{{{row['Inc']}}}{{(1+{rate})^{{{row['Year']}}}}}" for _, row in ed_df.iterrows() if row['Inc']>0])
    ic_val = ed_df.iloc[0]['Inv']
    npv_sub = f"NPV = ({inc_part}) - {fmt % ic_val}"

if m2['NPV'] > 0:
    npv_an = f"""
    <b>Результат положительный (Проект выгоден).</b><br>
    Значение NPV = {fmt % m2['NPV']} (больше 0).<br>
    Это значит, что проект не только возвращает все вложенные деньги, но и <b>приносит чистую прибыль</b> сверху.
    """
    st_npv = "success"
else:
    npv_an = f"""
    <b>Результат отрицательный (Невыгодно).</b><br>
    Значение NPV = {fmt % m2['NPV']} (меньше 0).<br>
    Денег проект принесет меньше, чем вы потратите (с учетом их стоимости). Вы уйдете в минус.
    """
    st_npv = "error"

conclusion_box("💰 NPV", fmt % m2['NPV'], npv_an, 
               [{'type':'latex', 'content': npv_f}, {'type':'latex', 'content': npv_sub}], st_npv)

# 2. PI
if m2['Is_Dist']:
    pi_f = r"PI = \frac{\sum P_k / (1+r)^k}{\sum IC_k / (1+r)^{k-1}}"
else:
    pi_f = r"PI = \frac{\sum P_k / (1+r)^k}{IC}"

pi_numer = " + ".join([f"\\frac{{{row['Inc']}}}{{(1+{rate})^{{{row['Year']}}}}}" for _, row in ed_df.iterrows() if row['Inc']>0])
if m2['Is_Dist']:
    pi_denom = " + ".join([f"\\frac{{{row['Inv']}}}{{(1+{rate})^{{{row['Year']-1}}}}}" for _, row in ed_df.iterrows() if row['Inv']>0])
else:
    pi_denom = f"{fmt % ed_df.iloc[0]['Inv']}"

pi_sub = f"PI = \\frac{{{pi_numer}}}{{{pi_denom}}} = \\frac{{{fmt % m2['PV_In']}}}{{{fmt % m2['PV_Out']}}}"

if m2['PI'] > 1:
    pi_an = f"""
    <b>Эффективно.</b><br>
    Индекс {fmt % m2['PI']} > 1.<br>
    На каждый вложенный рубль проект возвращает <b>{fmt % m2['PI']} руб.</b> дохода. Мы в плюсе.
    """
    st_pi = "success"
else:
    pi_an = f"""
    <b>Неэффективно.</b><br>
    Индекс {fmt % m2['PI']} < 1.<br>
    Проект возвращает меньше денег, чем вы вкладываете (в пересчете на сегодняшние деньги).
    """
    st_pi = "error"

conclusion_box("📈 PI", fmt % m2['PI'], pi_an, 
               [{'type':'latex', 'content': pi_f}, {'type':'latex', 'content': pi_sub}], st_pi)

# 3. PP & DPP
with st.container(border=True):
    st.markdown('<div class="metric-title">⏳ Окупаемость</div>', unsafe_allow_html=True)
    c_pp, c_dpp = st.columns(2)
    
    with c_pp:
        st.subheader("PP (Простой)")
        val_pp = f"{fmt % m2['PP']} лет" if m2['PP'] else "—"
        st.markdown(f"**Результат: {val_pp}**")
        if m2['PP']:
            steps_txt = []
            cum = 0
            for k in range(m2['PP_Idx'] + 1):
                row = ed_df.iloc[k]
                yr = row['Year']
                # Details: Inc - Inv = Net
                inc = row['Inc']
                inv = row['Inv']
                net = inc - inv
                prev = cum
                cum += net
                
                step_desc = f"Год {yr}: Доход {fmt % inc} - Инвест {fmt % inv} = Поток {fmt % net}"
                step_bal = f"Баланс: {fmt % prev} + ({fmt % net}) = {fmt % cum}"
                
                steps_txt.append(step_desc)
                steps_txt.append(step_bal)
                steps_txt.append("-" * 20)
            
            with st.expander("Расчет потока", expanded=True):
                st.text("\n".join(steps_txt))
            
            idx = m2['PP_Idx']
            T = ed_df.iloc[idx-1]['Year']
            rem = abs(m2['Cum_Flow'][idx-1])
            nxt = m2['Net_Flows'][idx]
            st.latex(f"PP = {T} + \\frac{{|{fmt % -rem}|}}{{{fmt % nxt}}} = {fmt % m2['PP']}")
            
            if m2['PP'] <= years_p:
               st.success(f"✅ Проект окупается за {val_pp}.")
               st.markdown("Мы вернем свои деньги достаточно быстро (раньше, чем проект закончится).")
            else:
               st.error("❌ Долго окупается.")
               st.markdown("Проект закончится раньше, чем мы успеем вернуть деньги без учета инфляции.")

    with c_dpp:
        st.subheader("DPP (Дисконтированный)")
        val_dpp = f"{fmt % m2['DPP']} лет" if m2['DPP'] else "—"
        st.markdown(f"**Результат: {val_dpp}**")
        if m2['DPP']:
            dcf_steps = []
            cum = 0
            for k in range(m2['DPP_Idx'] + 1):
                row = ed_df.iloc[k]
                yr = row['Year']
                inc = row['Inc']
                inv = row['Inv']
                net = inc - inv
                dcf = m2['DCF_Series'][k]
                
                step_1 = f"1. Дисконт: Поток({fmt % net}) / (1+{rate})^{yr} = {fmt % dcf}"
                prev = cum
                cum += dcf
                step_2 = f"2. Баланс: {fmt % prev} + {fmt % dcf} = {fmt % cum}"
                
                dcf_steps.append(f"--- Год {yr} ---")
                dcf_steps.append(step_1)
                dcf_steps.append(step_2)
                
            with st.expander("Расчет DCF", expanded=True):
                 st.text("\n".join(dcf_steps))
            
            idx = m2['DPP_Idx']
            T = ed_df.iloc[idx-1]['Year']
            rem = abs(m2['Cum_DCF'][idx-1])
            nxt = m2['DCF_Series'][idx]
            st.latex(f"DPP = {T} + \\frac{{|{fmt % -rem}|}}{{{fmt % nxt}}} = {fmt % m2['DPP']}")
            
            if m2['DPP'] <= years_p:
               st.success(f"✅ Проект окупается за {val_dpp} (реально).")
               st.markdown(f"Даже с учетом того, что деньги обесцениваются со ставкой {rate_pc}%, мы всё равно выйдем в плюс на {int(m2['DPP'])+1}-й год.")
            else:
               st.error("❌ Не окупается (с учетом ставки).")
               st.markdown("Из-за обесценивания денег (дисконта) проект не успевает вернуть вложения.")

# 4. IRR
irr_an = f"IRR = {fmt % m2['IRR']}%." if m2['IRR'] else "—"
if m2['IRR'] and m2['IRR'] > rate_pc:
    irr_an += f"""<br><br>
    <b>Проект надежен.</b><br>
    Максимальная ставка кредита или инфляции, которую выдержит проект — <b>{fmt % m2['IRR']}%</b>.
    Это выше вашей текущей ставки ({rate_pc}%), так что запас прочности есть.
    """
    st_irr = "success"
else:
    irr_an += f"""<br><br>
    <b>Слишком рискованно.</b><br>
    Проект приносит доходность всего {fmt % m2['IRR']}%, а вы требуете {rate_pc}%. 
    Лучше положить деньги в банк или найти другой проект.
    """
    st_irr = "error"

ir_cols = st.columns(2)
d1 = int(m2['IRR'] // 1) if m2['IRR'] else 10
r1_int = ir_cols[0].number_input("r1 (целое)", value=d1, step=1, format="%d")
r2_int = ir_cols[1].number_input("r2 (целое)", value=d1+5, step=1, format="%d")

r1_dec, r2_dec = r1_int/100, r2_int/100
n1 = calc_npv_manual_viz(r1_dec, ed_df, m2['Is_Dist'])
n2 = calc_npv_manual_viz(r2_dec, ed_df, m2['Is_Dist'])

irr_vis_1 = r"IRR \approx r_{(+)} + \frac{NPV_{(+)}}{NPV_{(+)} - NPV_{(-)}} \cdot (r_{(-)} - r_{(+)})"
irr_vis_2 = f"IRR \\approx {r1_dec} + \\frac{{{fmt % n1}}}{{{fmt % n1} - ({fmt % n2})}} \\cdot ({r2_dec} - {r1_dec})"

conclusion_box("🚀 IRR", f"{fmt % m2['IRR']}%" if m2['IRR'] else "—", irr_an, 
               [{'type':'latex', 'content': irr_vis_1}, {'type':'latex', 'content': irr_vis_2}], st_irr)
