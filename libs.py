#Librerias descargadas
# Todos los imports del proyecto se centralizan aquí.

# Estándar
import argparse
import sys
import warnings
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

# Datos y cálculo
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

# Optimización de portafolios
from pypfopt import (
    EfficientCVaR,
    EfficientFrontier,
    expected_returns,
    objective_functions,
    risk_models,
)
from pypfopt.discrete_allocation import DiscreteAllocation

# Visualización e interactividad
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ipywidgets as widgets
from IPython.display import Markdown, display

# Silenciar avisos internos de yfinance/pypfopt que no afectan los resultados
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pypfopt")

print("Librerias descargadas correctamente")
