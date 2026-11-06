#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño de dique vertical (cajón de hormigón)
Métodos de Goda (1974) y Takahashi (1994)
Autores base: M. Clavero, R. Bergillos, M. Ortega  |  GDFA · IISTA · UGR · 2018-2025
Versión web (Streamlit): M. Clavero Gilabert — 2026
"""

from __future__ import division

import math

import numpy as np
import streamlit as st
from scipy.optimize import fsolve


# ═══════════════════════════════════════════════════════════════════
#  FUNCIONES DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════


def wavnum(t, h):
    """Número de onda — resuelve σ²/g = k·tanh(k·h) con fsolve."""
    g = 9.81
    gamma_v = (2 * np.pi / t) ** 2.0 * h / g
    sigma2 = (2 * np.pi / t) ** 2

    def func(k):
        return sigma2 / g - k * np.tanh(k * h)

    ko = np.sqrt(gamma_v / h) if gamma_v > np.pi ** 2 else gamma_v / h
    k = fsolve(func, ko)
    return float(abs(k[0]))


def Goda(theta1, Hc, h, Hsp, tanfondo, L1, d, hp, gammaw, Fc):
    beta = max(0.0, theta1 - 15 * np.pi / 180)
    eta   = 0.75 * (1 + math.cos(beta)) * Hc
    hb    = h + 5 * Hsp * tanfondo
    alpha1 = 0.6 + 0.5 * ((4 * np.pi * h / L1) / math.sinh(4 * np.pi * h / L1)) ** 2
    alpha2 = min(((hb - d) / 3 / hb) * (Hc / d) ** 2, 2 * d / Hc)
    alpha3 = 1 - (hp / h) * (1 - 1 / math.cosh(2 * np.pi * h / L1))
    p1 = 0.5 * (1 + math.cos(beta)) * (alpha1 + alpha2 * math.cos(beta) ** 2) * gammaw * Hc
    p3 = alpha3 * p1
    p4 = (1 - Fc / eta) * p1 if eta > Fc else 0.0
    pu = 0.5 * (1 + math.cos(beta)) * alpha1 * alpha3 * gammaw * Hc
    return p1, p3, p4, pu


def Takahashi(theta1, Hc, h, L1, Hsp, tanfondo, d, hp, Fc, Bm, gammaw, k1):
    hb = h + 5 * Hsp * tanfondo
    beta = max(0.0, theta1 - 15 * np.pi / 180)
    lam1 = lam2 = lam3 = 1.0
    eta = 0.75 * (1 + math.cos(beta)) * Hc * lam1
    alpha1 = 0.6 + 0.5 * ((4 * np.pi * h / L1) / math.sinh(4 * np.pi * h / L1)) ** 2
    alpha2 = min(((hb - d) / (3 * hb)) * (Hc / d) ** 2, 2 * d / Hc)
    alpha3 = 1 - hp * (1 - 1 / math.cosh(2 * np.pi * h / L1)) / h
    hc_ast = min(eta, Fc)
    alpha4 = 1 - hc_ast / eta
    d11 = 0.93 * (Bm / L1 - 0.12) + 0.36 * ((h - d) / h - 0.6)
    d22 = -0.36 * (Bm / L1 - 0.12) + 0.93 * ((h - d) / h - 0.6)
    delta1 = (20 if d11 <= 0 else 15) * d11
    delta2 = (4.9 if d22 <= 0 else 3.0) * d22
    aI0 = Hc / d if Hc <= 2 * d else 2.0
    aI1 = (math.cos(delta2) / math.cosh(delta1) if delta2 <= 0
           else 1.0 / (math.cosh(delta1) * math.sqrt(math.cosh(delta2))))
    alpha_ast = max(alpha2, aI0 * aI1)
    p1 = 0.5 * (1 + math.cos(beta)) * (alpha1 * lam1 + alpha_ast * lam2 * math.cos(beta) ** 2) * gammaw * Hc
    p2 = p1 / math.cosh(k1 * h)
    p3 = alpha3 * p1
    p4 = alpha4 * p1
    pu = 0.5 * (1 + math.cos(beta)) * alpha1 * alpha3 * lam3 * gammaw * Hc
    return p1, p2, p3, p4, pu


def calcular_propagacion(params):
    """Paso previo: propagación y condición de rotura (sin necesitar tpH)."""
    T        = params["T"]
    h        = params["h"]
    Bb       = params["Bb"]
    S1       = params["S1"]
    S2       = params["S2"]
    fuente   = params.get("fuente_hs", "simar")

    d  = h - S1 - S2
    hp = h - S1

    g      = 9.81
    sigma  = 2 * math.pi / T
    k1     = wavnum(T, h)
    L1     = 2 * math.pi / k1

    if fuente == "pie_dique":
        Hsp    = params["Hsp_directo"]
        Ks     = 1.0
        Kr     = 1.0
    else:
        Hs0    = params["Hs0"]
        theta0 = params.get("theta0", 0.0)
        L0     = g * T ** 2 / (2 * math.pi)
        k0     = 2 * math.pi / L0
        theta1 = math.asin(k0 * math.sin(math.radians(theta0)) / k1)
        Kr     = math.sqrt(math.cos(math.radians(theta0)) / math.cos(theta1))
        cg0    = 0.5 * L0 / T
        cg1    = 0.5 * sigma / k1 * (1 + 2 * k1 * h / math.sinh(2 * k1 * h))
        Ks     = math.sqrt(cg0 / cg1)
        Hsp    = Hs0 * Ks * Kr

    R           = 1
    Hmax        = 1.8 * Hsp
    Hrot_crit   = (0.11 + 0.03 * (1 - R) / (1 + R)) * math.tanh(k1 * h)
    rompe_fondo = (Hmax / L1) > Hrot_crit
    Coef_a      = (h - d) / h
    Coef_b      = Bb / L1
    rompe_berma = (not rompe_fondo) and Coef_a >= 0.3 and Coef_b >= 0.01

    return {
        "Hsp": Hsp, "L1": L1, "Hmax": Hmax,
        "d": d, "hp": hp, "dh": d / h,
        "Ks": Ks, "Kr": Kr,
        "rompe_fondo": rompe_fondo, "rompe_berma": rompe_berma,
        "Coef_a": Coef_a, "Coef_b": Coef_b,
    }


def calcular(params):
    """Cálculo completo del dique vertical."""
    gammaw   = params["gammaw"]
    gammas   = params["gammas"]
    gammae   = params["gammae"]
    T        = params["T"]
    h        = params["h"]
    Bb       = params["Bb"]
    S1       = params["S1"]
    S2       = params["S2"]
    tanfondo = params["tanfondo"]
    Fc       = params["Fc"]
    Fm       = params["Fm"]
    botaolas = int(params["botaolas"])
    CS       = params["CS"]
    mu       = params["mu"]
    tpH      = params["tpH"]
    metodo_f = params.get("metodo_forzado", "auto")
    fuente   = params.get("fuente_hs", "simar")

    d  = h - S1 - S2
    hp = h - S1

    if d <= 0:
        raise ValueError(f"d = h − S1 − S2 = {d:.3f} m ≤ 0. Revisa S1, S2 y h.")
    if hp <= 0:
        raise ValueError(f"hp = h − S1 = {hp:.3f} m ≤ 0. Revisa S1 y h.")

    g      = 9.81
    sigma  = 2 * math.pi / T
    k1     = wavnum(T, h)
    L1     = 2 * math.pi / k1

    if fuente == "pie_dique":
        Hsp    = params["Hsp_directo"]
        Ks     = 1.0
        Kr     = 1.0
        theta1 = 0.0
    else:
        Hs0    = params["Hs0"]
        theta0 = params.get("theta0", 0.0)
        L0     = g * T ** 2 / (2 * math.pi)
        k0     = 2 * math.pi / L0
        theta1 = math.asin(k0 * math.sin(math.radians(theta0)) / k1)
        Kr     = math.sqrt(math.cos(math.radians(theta0)) / math.cos(theta1))
        cg0    = 0.5 * L0 / T
        cg1    = 0.5 * sigma / k1 * (1 + 2 * k1 * h / math.sinh(2 * k1 * h))
        Ks     = math.sqrt(cg0 / cg1)
        Hsp    = Hs0 * Ks * Kr

    R           = 1
    Hmax        = 1.8 * Hsp
    Hrot_crit   = (0.11 + 0.03 * (1 - R) / (1 + R)) * math.tanh(k1 * h)
    rompe_fondo = (Hmax / L1) > Hrot_crit
    Coef_a      = (h - d) / h
    Coef_b      = Bb / L1
    rompe_berma = (not rompe_fondo) and Coef_a >= 0.3 and Coef_b >= 0.01
    dh          = d / h

    # ── Selección del método ──
    if metodo_f == "goda":
        metodo_usado = "Goda (1974)  [forzado]"
        Hc = Hmax
    elif metodo_f == "takahashi":
        metodo_usado = "Takahashi (1994)  [forzado]"
        Hc = 0.11 * math.tanh(k1 * h) * L1
    else:
        if rompe_fondo:
            metodo_usado = "Takahashi (1994)  — rotura por fondo"
            Hc = 0.11 * math.tanh(k1 * h) * L1
        elif rompe_berma:
            metodo_usado = "Takahashi (1994)  — rotura sobre berma"
            Hc = 0.11 * math.tanh(k1 * h) * L1
        else:
            metodo_usado = "Goda (1974)  — sin rotura"
            Hc = Hmax

    # ── Presiones ──
    if "Goda" in metodo_usado:
        p1, p3, p4, pu = Goda(theta1, Hc, h, Hsp, tanfondo, L1, d, hp, gammaw, Fc)
        p2 = None
    else:
        p1, p2, p3, p4, pu = Takahashi(
            theta1, Hc, h, L1, Hsp, tanfondo, d, hp, Fc, Bb, gammaw, k1
        )

    # ── Fuerza horizontal ──
    Fh = 0.5 * (p1 + p3) * hp + 0.5 * (p1 + p4) * Fc

    # ── Deslizamiento ──
    B1_des = CS * Fh / mu
    B2_des = (
        (1 / botaolas) * (Fc - Fm) * gammas
        + Fm * gammas
        + hp * (gammas - gammaw)
        - 0.5 * pu
    )
    if abs(B2_des) < 1e-10:
        raise ValueError("Denominador nulo en deslizamiento — revisa Fc, Fm, gammas y pu.")
    B_des = B1_des / B2_des

    # ── Momentos y vuelco ──
    Mh = (
        0.5 * (p1 - p3) * (2 / 3) * hp ** 2
        + 0.5 * p3 * hp ** 2
        + 0.5 * (p1 - p4) * Fc * (hp + Fc / 3)
        + p4 * Fc * (hp + Fc / 2)
    )
    B1_vuelco = Mh * CS
    B2_vuelco = (
        (1 / botaolas) * (Fc - Fm) * gammas * (1 - 1 / (2 * botaolas))
        + Fm * gammas * 0.5
        + hp * (gammas - gammaw) * 0.5
        - CS * 0.5 * pu * (2 / 3)
    )
    if abs(B2_vuelco) < 1e-10:
        raise ValueError("Denominador nulo en vuelco — revisa parámetros del cajón.")
    ratio_v = B1_vuelco / B2_vuelco
    if ratio_v < 0:
        raise ValueError(f"Ratio de vuelco negativo ({ratio_v:.4f}). Comprueba Fc, Fm, gammas y CS.")
    B_vuelco = math.sqrt(ratio_v)
    Bmax     = max(B_des, B_vuelco)

    # ── Bloque de guarda ──
    tp = tpH * Hsp

    # ── Berma de pie ──
    metodo_berma = params.get("metodo_berma", "tanimoto")
    Nod          = params.get("Nod", 0.5)

    if metodo_berma == "madrigal":
        kappa  = None
        _factor = 5.8 * dh - 0.6
        if _factor <= 0:
            raise ValueError(
                f"Madrigal y Valdés: factor (5.8·d/h − 0.6) = {_factor:.4f} ≤ 0. "
                "Revisa la relación d/h."
            )
        Ns  = _factor * (Nod ** 0.19)
        Delta = gammae / gammaw - 1
        Db  = Hsp / (Ns * Delta)
        Wb  = gammae * Db ** 3
    else:
        kappa1 = (2 * k1 * hp) / np.sinh(2 * k1 * hp)
        kappa2 = np.max([
            0.45 * np.sin(theta1) ** 2 * np.cos(k1 * Bb * np.cos(theta1)) ** 2,
            np.cos(theta1) ** 2 * np.sin(k1 * Bb * np.cos(theta1)) ** 2,
        ])
        kappa  = kappa1 * kappa2
        kap_cb = kappa ** (1 / 3)
        if kap_cb < 1e-10:
            raise ValueError("κ ≈ 0: parámetros fuera del rango de Tanimoto.")
        Ns = np.max([
            1.8,
            1.3 * ((1 - kappa) * hp) / (kap_cb * Hsp)
            + 1.8 * np.exp(-1.5 * (1 - kappa) ** 2 * hp / (kap_cb * Hsp)),
        ])
        Db = Hsp / (1.65 * Ns)
        Wb = gammae * Db ** 3

    # ── Fuerza vertical (empuje ascendente) y momento vertical ──
    Fu = 0.5 * pu * Bmax                   # empuje ascendente (distribución triangular)
    Mu = Fu * (2.0 / 3.0) * Bmax          # momento de Fu respecto al talón interior

    nota = None  # Sin rotura → Goda (correcto); con rotura → Takahashi

    return {
        "Hsp": Hsp, "L1": L1, "Hmax": Hmax, "Hc": Hc,
        "d": d, "hp": hp, "dh": dh,
        "Ks": Ks, "Kr": Kr,
        "rompe_fondo": rompe_fondo, "rompe_berma": rompe_berma,
        "Coef_a": Coef_a, "Coef_b": Coef_b,
        "metodo_usado": metodo_usado,
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "pu": pu,
        "Fh": Fh, "Mh": Mh, "Fu": Fu, "Mu": Mu,
        "B_des": B_des, "B_vuelco": B_vuelco, "Bmax": Bmax,
        "tp": tp, "tpH": tpH,
        "kappa": kappa, "Ns": Ns, "Db": Db, "Wb": Wb,
        "metodo_berma": metodo_berma, "Nod": Nod,
        "nota": nota,
    }


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dique Vertical (Cajón) — GDFA·UGR",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d2b55;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown {
    color: #e8edf4 !important;
}
[data-testid="stSidebar"] .stExpander {
    background-color: #142f5c !important;
    border: 1px solid #1a4a8a !important;
    border-radius: 8px;
    margin-bottom: 6px;
}
[data-testid="stSidebar"] .stExpander summary {
    color: #ffffff !important;
    font-weight: 600;
}
[data-testid="stSidebar"] details[open],
[data-testid="stSidebar"] details[open] > summary,
[data-testid="stSidebar"] details[open] > div,
[data-testid="stSidebar"] details[open] .streamlit-expanderContent,
[data-testid="stSidebar"] details[open] .st-expander-content {
    background-color: #162d58 !important;
}
[data-testid="stSidebar"] details[open] > div * {
    background-color: transparent !important;
}
[data-testid="stSidebar"] details[open] input {
    background-color: #122648 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background-color: #1a3a6b !important;
    color: #ffffff !important;
    border: 1px solid #3a6aab !important;
}
/* ── Cabecera principal ── */
.app-header {
    background: linear-gradient(135deg, #0d2b55 0%, #1a4a8a 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.app-header h1 { color: white !important; margin: 0; font-size: 1.6rem; }
.app-header p  { color: #d0d8e4 !important; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
/* ── Tarjetas de resultados ── */
.metric-card {
    background: white;
    border: 1px solid #d0d8e4;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(13,43,85,.08);
}
/* ── Tabla de resultados ── */
table { width: 100%; border-collapse: collapse; }
th { background: #0d2b55; color: white; padding: 8px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e8edf4; }
tr:hover td { background: #f2f4f7; }
/* ── Botón principal ── */
div[data-testid="stButton"] > button {
    background-color: #0d2b55;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: .05em;
    width: 100%;
    transition: background .2s;
}
div[data-testid="stButton"] > button:hover {
    background-color: #1a4a8a;
}
/* ── Tabla resumen: título del expander a 20px ── */
[data-testid="stMain"] [data-testid="stExpander"] summary p,
[data-testid="stMain"] [data-testid="stExpander"] summary span {
    font-size: 20px !important;
}
/* ── Ocultar botones +/- de los campos numéricos ── */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    display: none;
}
/* ── Radio buttons en sidebar: círculo azul claro ── */
[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #7eb8f7 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] span {
    border-color: #7eb8f7 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] span {
    background-color: #7eb8f7 !important;
    border-color: #7eb8f7 !important;
}
[data-testid="stSidebar"] .st-c3,
[data-testid="stSidebar"] .st-c4,
[data-testid="stSidebar"] .st-c5 {
    border-color: #7eb8f7 !important;
}
[data-testid="stSidebar"] [role="radio"] > div:first-child {
    border-color: #7eb8f7 !important;
    background-color: transparent !important;
}
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] > div:first-child {
    background-color: #7eb8f7 !important;
    border-color: #7eb8f7 !important;
}
[data-testid="stSidebar"] .stRadio label > div:first-child {
    border-color: #7eb8f7 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  CABECERA
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <h1>DIMENSIONAMIENTO DE DIQUE VERTICAL</h1>
  <p>M. Clavero &nbsp;·&nbsp; GDFA · IISTA · UGR</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  BARRA LATERAL — DATOS DE ENTRADA
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📋 DATOS DE ENTRADA")

    # ── 1. Geometría y materiales ─────────────────────────────────
    with st.expander("Geometría y materiales", expanded=False):
        gammaw = st.number_input("γ agua (t/m³)",              value=1.025, step=0.001, format="%.3f", key="gammaw")
        gammas = st.number_input("γ hormigón (t/m³)",          value=2.35,  step=0.01,  format="%.3f", key="gammas")
        gammae = st.number_input("γ piedra escollera (t/m³)",  value=2.65,  step=0.01,  format="%.3f", key="gammae")
        Bb     = st.number_input("Anchura cimentación Bb (m)", value=8.0,   step=0.5,   format="%.2f", key="Bb")
        S1     = st.number_input("Espesor cimentación S1 (m)", value=1.8,   step=0.1,   format="%.2f", key="S1")
        S2     = st.number_input("Espesor berma de pie S2 (m)",value=3.2,   step=0.1,   format="%.2f", key="S2")

    # ── 2. Oleaje y emplazamiento ──────────────────────────────────
    with st.expander("Condiciones de oleaje y emplazamiento", expanded=False):
        fuente_hs = st.radio(
            "Punto de referencia de Hs",
            options=["simar", "pie_dique"],
            format_func=lambda x: {
                "simar":     "Hs en SIMAR / aguas profundas",
                "pie_dique": "Hs a pie de dique",
            }[x],
            index=0,
            key="fuente_hs",
            horizontal=True,
        )
        # Siempre se definen ambas variables; el selector controla cuál se muestra
        Hs0         = 5.2
        theta0      = 0.0
        Hsp_directo = 5.2
        if fuente_hs == "simar":
            Hs0    = st.number_input("Hs₀ en SIMAR (m)",         value=5.2, step=0.1, format="%.3f", key="Hs0")
            theta0 = st.number_input("Ángulo incidencia θ₀ (°)", value=0.0, step=1.0, format="%.1f", key="theta0")
        else:
            Hsp_directo = st.number_input("Hsp a pie de dique (m)", value=5.2, step=0.1, format="%.3f", key="Hsp_directo")
        T        = st.number_input("Periodo T (s)",             value=11.0,         step=0.5,   format="%.2f", key="T")
        h        = st.number_input("Calado pie de dique h (m)", value=35.0,         step=0.5,   format="%.2f", key="h")
        tanfondo = st.number_input("Pendiente de fondo  tan α", value=round(1/60,5),step=0.001, format="%.5f", key="tanfondo")

    # ── Hsp previo para Franco & Franco (calculado antes de la sección) ──
    try:
        _ff_p = {
            "Hs0": Hs0, "T": T, "theta0": theta0, "h": h,
            "Bb": Bb, "S1": S1, "S2": S2,
            "fuente_hs": fuente_hs, "Hsp_directo": Hsp_directo,
        }
        _Hsp_prev = calcular_propagacion(_ff_p)["Hsp"]
    except Exception:
        _Hsp_prev = None

    # ── 3. Superestructura ────────────────────────────────────────
    with st.expander("Superestructura", expanded=False):
        fc_modo = st.radio(
            "Cota de francobordo Fc",
            options=["directo", "franco"],
            format_func=lambda x: {
                "directo": "Introducir Fc directamente",
                "franco":  "Calcular — Franco y Franco (1999)",
            }[x],
            index=0,
            key="fc_modo",
            horizontal=True,
        )
        if fc_modo == "directo":
            Fc = st.number_input("Cota de francobordo Fc (m)", value=10.0, step=0.5, format="%.2f", key="Fc")
        else:
            q_ff = st.number_input(
                "Caudal medio de rebase q (m³/s/ml)",
                value=0.001, min_value=1e-9, step=0.0001, format="%.6f",
                key="q_ff",
                help="Valores orientativos: paseo peatonal ≈ 0.0001; zona portuaria ≈ 0.001–0.01 m³/s/ml",
            )
            if _Hsp_prev is not None and _Hsp_prev > 0:
                _arg = q_ff / (0.082 * math.sqrt(9.81 * _Hsp_prev ** 3))
                if 0 < _arg < 1:
                    Fc = -(_Hsp_prev / 3.0) * math.log(_arg)
                    st.info(
                        f"**Fc = {Fc:.3f} m**  "
                        f"(Hsp = {_Hsp_prev:.3f} m · q = {q_ff:.6f} m³/s/ml)"
                    )
                else:
                    st.warning("El valor de q está fuera del rango de la fórmula. Comprueba los datos de oleaje.")
                    Fc = 10.0
            else:
                st.warning("Introduce primero los datos de oleaje para calcular Fc.")
                Fc = 10.0
        Fm       = st.number_input("Cota muelle Fm (m)",          value=3.0,  step=0.5,  format="%.2f", key="Fm")
        CS       = st.number_input("Coef. seguridad CS",          value=1.2,  step=0.05, format="%.2f", key="CS")
        mu       = st.number_input("Coef. rozamiento μ",          value=0.6,  step=0.05, format="%.3f", key="mu")
        botaolas = st.number_input("Divisor anchura B/n cajones", value=3,    step=1, min_value=1,      key="botaolas")

    # ── 4. Cimentación (bloque de guarda) ─────────────────────────
    with st.expander("Cimentación", expanded=False):
        # -- Preview automático: mostrar d/h como dato conocido --
        try:
            _cim_p = {
                "Hs0": Hs0, "T": T, "theta0": theta0, "h": h,
                "Bb": Bb, "S1": S1, "S2": S2,
                "fuente_hs": fuente_hs, "Hsp_directo": Hsp_directo,
            }
            _cim_pr = calcular_propagacion(_cim_p)
            _cim_dh  = _cim_pr["dh"]
            _cim_d   = _cim_pr["d"]
            _cim_h   = h
            st.info(
                f"**d/h = {_cim_dh:.4f}**  "
                f"(d = {_cim_d:.3f} m · h = {_cim_h:.3f} m)  —  "
                f"utiliza este valor para leer tp/H del ábaco del bloque de guarda."
            )
        except Exception:
            st.caption("Introduce los datos de oleaje para obtener d/h.")

        tpH = st.number_input(
            "tp / H  — del ábaco (bloque de guarda)",
            value=0.1, min_value=0.04, max_value=0.35, step=0.01, format="%.3f",
            help="Lee el valor de tp/H del ábaco de diseño usando el d/h calculado. Rango: 0.04 – 0.35",
            key="tpH",
        )

    # ── 5. Berma de pie ───────────────────────────────────────────
    with st.expander("Berma de pie", expanded=False):
        metodo_berma = st.radio(
            "Método de cálculo",
            options=["tanimoto", "madrigal"],
            format_func=lambda x: {
                "tanimoto": "Tanimoto et al. (1982)",
                "madrigal": "Madrigal y Valdés (1995)",
            }[x],
            index=1,
            key="metodo_berma",
            horizontal=True,
        )
        if metodo_berma == "madrigal":
            Nod = st.number_input(
                "N_od  (nº de piezas dañadas por metro lineal)",
                value=0.5, min_value=0.0, max_value=5.0, step=0.1, format="%.2f",
                help="Parámetro de daño: 0.5 = inicio de daño (valor habitual de diseño).",
                key="Nod",
            )
        else:
            Nod = 0.5

    # ── 6. Método de cálculo de presiones  (con preview automático) ──
    with st.expander("Método de cálculo de presiones", expanded=False):
        # -- Preview automático --
        try:
            _p = {
                "Hs0": Hs0, "T": T, "theta0": theta0, "h": h,
                "Bb": Bb, "S1": S1, "S2": S2,
                "fuente_hs": fuente_hs, "Hsp_directo": Hsp_directo,
            }
            _pr = calcular_propagacion(_p)
            _dh   = _pr["dh"]
            _Hsp  = _pr["Hsp"]
            _L1   = _pr["L1"]
            _d    = _pr["d"]
            _hp_v = _pr["hp"]

            if _pr["rompe_fondo"]:
                _rot_txt = "⚠️ Rotura por **fondo** → se aplicará Takahashi"
                _rot_color = "warning"
            elif _pr["rompe_berma"]:
                _rot_txt = (f"⚠️ Rotura sobre **berma**  "
                            f"(Coef_a = {_pr['Coef_a']:.3f} ≥ 0.3, "
                            f"Coef_b = {_pr['Coef_b']:.3f} ≥ 0.01) → Takahashi")
                _rot_color = "warning"
            else:
                _rot_txt = (f"✅ Sin rotura → se aplicará **Goda (1974)**  "
                            f"(Coef_a = {_pr['Coef_a']:.3f}, "
                            f"Coef_b = {_pr['Coef_b']:.3f})")
                _rot_color = "info"

            st.info(
                f"**Propagación automática**\n\n"
                f"Hsp = {_Hsp:.3f} m\n\n"
                f"L = {_L1:.2f} m\n\n"
                f"d = {_d:.2f} m\n\n"
                f"hp = {_hp_v:.2f} m"
            )
            st.info(f"**d/h = {_dh:.4f}**\n\n{_rot_txt}")

        except Exception as _e:
            st.warning(f"Preview no disponible: {_e}")

        st.divider()

        metodo_opt = st.radio(
            "Método de cálculo de presiones",
            options=["auto", "goda", "takahashi"],
            format_func=lambda x: {
                "auto":      "Automático  (según condición de rotura)",
                "goda":      "Forzar  Goda (1974)",
                "takahashi": "Forzar  Takahashi (1994)",
            }[x],
            index=0,
            key="metodo_opt",
        )


# ═══════════════════════════════════════════════════════════════════
#  ZONA PRINCIPAL — BOTÓN Y RESULTADOS
# ═══════════════════════════════════════════════════════════════════

col_btn, col_sp = st.columns([1, 3])
with col_btn:
    calcular_btn = st.button("⚙️  CALCULAR", use_container_width=True)

# ── Botón "Acerca de" fijo en esquina inferior derecha ──
st.markdown("""
<div style="position:fixed; bottom:24px; right:28px; z-index:9999;">
  <details>
    <summary style="
        background:#0d2b55; color:white;
        padding:8px 18px; border-radius:20px;
        cursor:pointer; list-style:none;
        font-weight:600; font-size:0.88rem;
        box-shadow:0 3px 10px rgba(0,0,0,.25);
        user-select:none;">
      ℹ️&nbsp; Acerca de
    </summary>
    <div style="
        background:white; border:1px solid #d0d8e4;
        border-radius:10px; padding:1rem 1.2rem;
        margin-top:8px; min-width:230px;
        box-shadow:0 6px 20px rgba(0,0,0,.15);
        font-size:0.9rem; line-height:1.7;
        position:relative; bottom:0; right:0;">
      <b style="color:#0d2b55;">Autores base</b><br>
      M. Clavero, R. Bergillos, M. Ortega<br>
      2018 – 2025<br><br>
      <b style="color:#0d2b55;">Integración web</b><br>
      M. Clavero Gilabert<br>
      2026<br><br>
      <span style="color:#6b7a8d;">GDFA · IISTA · UGR</span><br><br>
    </div>
  </details>
</div>
""", unsafe_allow_html=True)


if calcular_btn:
    params = {
        "gammaw": gammaw, "gammas": gammas, "gammae": gammae,
        "T": T, "h": h,
        "Bb": Bb, "S1": S1, "S2": S2, "tanfondo": tanfondo,
        "Fc": Fc, "Fm": Fm, "CS": CS, "mu": mu, "botaolas": int(botaolas),
        "tpH": tpH,
        "metodo_forzado": metodo_opt,
        "fuente_hs": fuente_hs,
        "Hs0": Hs0, "theta0": theta0, "Hsp_directo": Hsp_directo,
        "metodo_berma": metodo_berma, "Nod": Nod,
    }

    try:
        r = calcular(params)
    except ValueError as e:
        st.error(f"**Error de cálculo:** {e}")
        st.stop()
    except Exception as e:
        st.error(f"**Error inesperado:** {e}")
        st.stop()

    st.success("✅ Cálculo completado")
    st.markdown("---")

    # ══════════════════════════════════════════
    #  BLOQUE 1 — OLEAJE A PIE DE DIQUE
    # ══════════════════════════════════════════
    st.markdown("### Oleaje a pie de dique")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hsp pie de dique",     f"{r['Hsp']:.3f} m")
    c2.metric("Hmax de diseño",       f"{r['Hmax']:.3f} m",
              help="Hmax = 1.8·Hsp  (dique vertical)")
    c3.metric("Longitud de onda L",   f"{r['L1']:.2f} m")
    c4.metric("Rotura", "Fondo ⚠" if r["rompe_fondo"] else ("Berma ⚠" if r["rompe_berma"] else "No ✓"))

    st.caption(
        f"Propagación: Ks = {r['Ks']:.4f} · Kr = {r['Kr']:.4f} → "
        f"Hsp = Hs₀·Ks·Kr = {r['Hsp']:.3f} m · "
        f"d = {r['d']:.2f} m · hp = {r['hp']:.2f} m · d/h = {r['dh']:.4f}"
    )

    # ══════════════════════════════════════════
    #  BLOQUE 2 — MÉTODO Y PRESIONES
    # ══════════════════════════════════════════
    st.markdown("### Método y presiones")
    st.info(f"**Método aplicado:** {r['metodo_usado']}  ·  **Hc = {r['Hc']:.3f} m**")

    if not r["rompe_fondo"] and not r["rompe_berma"]:
        st.caption(
            f"Coef_a = (h−d)/h = {r['Coef_a']:.4f} · "
            f"Coef_b = Bb/L = {r['Coef_b']:.4f}"
        )

    if r.get("nota"):
        st.warning(r["nota"])

    # Presiones en columnas
    if r["p2"] is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("p₁  (t/m²)", f"{r['p1']:.4f}")
        c2.metric("p₂  (t/m²)", f"{r['p2']:.4f}", help="Solo Takahashi")
        c3.metric("p₃  (t/m²)", f"{r['p3']:.4f}")
        c4.metric("p₄  (t/m²)", f"{r['p4']:.4f}")
        c5.metric("pu  (t/m²)", f"{r['pu']:.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("p₁  (t/m²)", f"{r['p1']:.4f}")
        c2.metric("p₃  (t/m²)", f"{r['p3']:.4f}")
        c3.metric("p₄  (t/m²)", f"{r['p4']:.4f}")
        c4.metric("pu  (t/m²)", f"{r['pu']:.4f}")

    # ══════════════════════════════════════════
    #  BLOQUE 3 — FUERZAS Y MOMENTOS
    # ══════════════════════════════════════════
    st.markdown("### Fuerzas y momentos")
    c1, c2 = st.columns(2)
    c1.metric("Fuerza horizontal Fh  (t/m)",    f"{r['Fh']:.4f}")
    c2.metric("Momento horizontal Mh  (t·m/m)", f"{r['Mh']:.4f}")

    # ══════════════════════════════════════════
    #  BLOQUE 4 — ESTABILIDAD DEL CAJÓN
    # ══════════════════════════════════════════
    st.markdown("### Estabilidad del cajón")
    c1, c2, c3 = st.columns(3)
    c1.metric("B deslizamiento  (m)", f"{r['B_des']:.4f}")
    c2.metric("B vuelco  (m)",        f"{r['B_vuelco']:.4f}")
    c3.metric("**Bmax — anchura mínima cajón  (m)**", f"{r['Bmax']:.4f}")

    # ══════════════════════════════════════════
    #  BLOQUE 4 — BLOQUE DE GUARDA
    # ══════════════════════════════════════════
    st.markdown("### Bloque de guarda")
    c1, c2 = st.columns(2)
    c1.metric("d/h  →  tp/H del ábaco", f"{r['tpH']:.3f}")
    c2.metric("Espesor bloque de guarda tp  (m)", f"{r['tp']:.4f}")

    # ══════════════════════════════════════════
    #  BLOQUE 5 — BERMA DE PIE
    # ══════════════════════════════════════════
    _lbl_berma = ("Berma de pie  (Madrigal y Valdés 1995)"
                  if r["metodo_berma"] == "madrigal"
                  else "Berma de pie  (Tanimoto et al. 1982)")
    st.markdown(f"### {_lbl_berma}")
    if r["metodo_berma"] == "madrigal":
        c1, c2, c3 = st.columns(3)
        c1.metric("N_od",                          f"{r['Nod']:.2f}")
        c2.metric("Número estabilidad Ns",         f"{r['Ns']:.4f}")
        c3.metric("Espesor pieza Db  (m)",         f"{r['Db']:.4f}")
        st.metric("**Peso pieza Wb  (t)**",        f"{r['Wb']:.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coeficiente κ",             f"{r['kappa']:.6f}")
        c2.metric("Número estabilidad Ns",     f"{r['Ns']:.4f}")
        c3.metric("Espesor pieza Db  (m)",     f"{r['Db']:.4f}")
        c4.metric("**Peso pieza Wb  (t)**",    f"{r['Wb']:.4f}")

    # ══════════════════════════════════════════
    #  TABLA RESUMEN
    # ══════════════════════════════════════════
    with st.expander("📄 Tabla resumen de todos los resultados", expanded=False):
        import pandas as pd

        filas = [
            ("Oleaje",     "Hsp a pie de dique (m)",         f"{r['Hsp']:.4f}"),
            ("Oleaje",     "Hmax de diseño (m)",              f"{r['Hmax']:.4f}"),
            ("Oleaje",     "Hc (m)",                          f"{r['Hc']:.4f}"),
            ("Oleaje",     "Longitud de onda L (m)",          f"{r['L1']:.4f}"),
            ("Oleaje",     "Coef. shoaling Ks",               f"{r['Ks']:.6f}"),
            ("Oleaje",     "Coef. refracción Kr",             f"{r['Kr']:.6f}"),
            ("Oleaje",     "d (m)",                           f"{r['d']:.4f}"),
            ("Oleaje",     "hp (m)",                          f"{r['hp']:.4f}"),
            ("Oleaje",     "d/h",                             f"{r['dh']:.6f}"),
            ("Oleaje",     "Rotura por fondo",                "Sí" if r["rompe_fondo"] else "No"),
            ("Oleaje",     "Rotura sobre berma",              "Sí" if r["rompe_berma"] else "No"),
            ("Presiones",  "Método",                          r["metodo_usado"]),
            ("Presiones",  "p₁ (t/m²)",                      f"{r['p1']:.6f}"),
        ]
        if r["p2"] is not None:
            filas.append(("Presiones", "p₂ (t/m²)", f"{r['p2']:.6f}"))
        filas += [
            ("Presiones",  "p₃ (t/m²)",                      f"{r['p3']:.6f}"),
            ("Presiones",  "p₄ (t/m²)",                      f"{r['p4']:.6f}"),
            ("Presiones",  "pu (t/m²)",                       f"{r['pu']:.6f}"),
            ("Fuerzas y momentos", "Fh — fuerza horizontal (t/m)",    f"{r['Fh']:.6f}"),
            ("Fuerzas y momentos", "Mh — momento horizontal (t·m/m)", f"{r['Mh']:.6f}"),
            ("Estabilidad","B deslizamiento (m)",             f"{r['B_des']:.4f}"),
            ("Estabilidad","B vuelco (m)",                    f"{r['B_vuelco']:.4f}"),
            ("Estabilidad","Bmax — anchura mínima cajón (m)", f"{r['Bmax']:.4f}"),
            ("Bloque guarda","tp/H del ábaco",                f"{r['tpH']:.3f}"),
            ("Bloque guarda","tp espesor bloque (m)",         f"{r['tp']:.4f}"),
            *(
                [("Berma pie", "κ", f"{r['kappa']:.6f}")]
                if r["metodo_berma"] == "tanimoto" else
                [("Berma pie", "N_od", f"{r['Nod']:.2f}")]
            ),
            ("Berma pie",  "Método",                          "Madrigal y Valdés (1995)" if r["metodo_berma"] == "madrigal" else "Tanimoto et al. (1982)"),
            ("Berma pie",  "Ns",                              f"{r['Ns']:.4f}"),
            ("Berma pie",  "Db (m)",                          f"{r['Db']:.4f}"),
            ("Berma pie",  "Wb (t)",                          f"{r['Wb']:.4f}"),
        ]

        df_res = pd.DataFrame(filas, columns=["Bloque", "Parámetro", "Valor"])
        html_tabla = df_res.to_html(index=False, border=0)
        st.markdown(
            f'<div style="font-size:20px;">{html_tabla}</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════
    #  REFERENCIAS EMPLEADAS
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Referencias empleadas")

    st.markdown("**1. Presiones sobre el paramento**")
    st.markdown(
        "- Goda, Y. (1973). A new method of wave pressure calculation for the design of composite "
        "breakwaters. *Report of the Port and Harbour Research Institute*, 12(3):31–69.\n\n"
        "- Takahashi, S., Tanimoto, K., and Shimosako, K. (1994). A proposal of impulsive pressure "
        "coefficient for design of composite breakwaters. In *Proc. Int. Conf. Hydro-Tech. Eng. "
        "Port Harbor Constr. (Hydro-Port'94)*, pages 489–504. Port and Harbour Research Institute."
    )

    st.markdown("**2. Cota de francobordo**")
    st.markdown(
        "- Franco, L. and Franco, G. (1999). Overtopping of vertical and composite breakwaters "
        "under oblique and short-crested waves. *Coastal Engineering*, 36(3):235–263."
    )

    st.markdown("**3. Berma de pie y bloque de guarda**")
    st.markdown(
        "- Tanimoto, K., Yagyu, T., and Goda, Y. (1982). Irregular wave tests for composite "
        "breakwater foundations. In *Coastal Engineering 1982*, pages 2144–2163.\n\n"
        "- Madrigal, B. and Valdés, J. (1995). Study of rubble mound foundation stability. "
        "In *Proc. Final Workshop, MAST II, MCS-Project, 1995*.\n\n"
        "- U.S. Army Corps of Engineers (USACE) (2005). *Coastal Engineering Manual*. Reston, VA. "
        "Originalmente publicado por USACE en 2002; edición profesional ASCE en 2005."
    )

else:
    # ── Estado inicial ──
    st.info(
        "👈 **Introduce los datos en la barra lateral** y pulsa **⚙️ CALCULAR**.\n\n"
        "Las secciones están plegadas; haz clic en cada una para expandirla. "
        "La propagación de olas y la condición de rotura se calculan automáticamente "
        "al introducir los datos de oleaje — úsalos para leer tp/H del ábaco del bloque de guarda."
    )
    st.markdown("""
---
#### Flujo de trabajo recomendado

1. **Pesos específicos** — γ agua, hormigón, escollera.
2. **Condiciones de oleaje** — Hs₀ en SIMAR, periodo, ángulo y calado.
3. **Geometría** — anchura berma Bb, espesores S1 y S2, pendiente fondo.
4. **Superestructura** — cotas Fc y Fm, coeficientes CS, μ y divisor de cajones.
5. **Método y bloque de guarda** — el d/h se muestra automáticamente; lee tp/H del ábaco.
6. Pulsa **⚙️ CALCULAR**.
""")
