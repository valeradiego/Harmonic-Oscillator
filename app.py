
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from math import factorial

st.set_page_config(page_title="Quantum Harmonic Oscillator", layout="wide")

st.title("Quantum Harmonic Oscillator Time Evolution")
st.subheader("Diego Alejandro Valera Contreras")

case = st.selectbox(
    "Choose state",
    [
        "Energy eigenstate ψₙ(x,t)",
        "Superposition ψₙ,ₘ(x,t)",
        "Coherent state"
    ]
)

x0 = st.slider("x₀", 0.2, 3.0, 1.0)
omega = st.slider("ω", 0.1, 5.0, 1.0)

n = st.slider("n", 0, 10, 0)

if "Superposition" in case:
    m = st.slider("m", 0, 10, 1)

if "Coherent" in case:
    mean_n = st.slider("<n>", 0.0, 20.0, 5.0)


def ground_wave_function(x):
    return np.pi**(-0.25)/np.sqrt(x0)*np.exp(-x*x/(2*x0*x0))


def energy_eigenfunction(x,n):
    return eval_hermite(n,x/x0)/np.sqrt(2**n*factorial(n))*ground_wave_function(x)


def psi_n(x,t,n):
    return np.exp(-1j*omega*(n+0.5)*t)*energy_eigenfunction(x,n)


def psi_nm(x,t,n,m):
    return (psi_n(x,t,n)+psi_n(x,t,m))/np.sqrt(2)


def coherent_state(x,t):
    alpha=np.sqrt(mean_n)
    return (
        np.exp(-1j*omega*t/2)/(np.pi**0.25*np.sqrt(x0))
        *
        np.exp(
            -x*x/(2*x0*x0)
            +np.sqrt(2)*alpha*np.exp(-1j*omega*t)*x/x0
            -alpha**2*np.exp(-2j*omega*t)/2
            -mean_n/2
        )
    )


def state(x,t):
    if case.startswith("Energy"):
        return psi_n(x,t,n)
    elif case.startswith("Superposition"):
        return psi_nm(x,t,n,m)
    return coherent_state(x,t)


if st.button("Generate animation"):

    x=np.linspace(-6*x0,6*x0,800)

    placeholder=st.empty()

    for frame in range(200):
        t=frame/10

        psi=state(x,t)

        fig,ax=plt.subplots(3,1,figsize=(8,8))

        ax[0].plot(x,psi.real)
        ax[0].set_ylabel("Re ψ")

        ax[1].plot(x,psi.imag)
        ax[1].set_ylabel("Im ψ")

        ax[2].plot(x,np.abs(psi)**2)
        ax[2].set_ylabel("|ψ|²")
        ax[2].set_xlabel("x")

        fig.suptitle(f"t = {t:.2f} s")

        for a in ax:
            a.grid()

        placeholder.pyplot(fig)
        plt.close(fig)
