import subprocess

def install(package):
  subprocess.check_call(["pip", "install", package])

# Lista das bibliotecas necessárias (substitua por suas bibliotecas)
required_packages = ['pandas', 'numpy', 'streamlit', 'plotly', 'requests']

# Instala as bibliotecas
for package in required_packages:
  install(package)

# Executa o script Streamlit
subprocess.call(['streamlit', 'run', 'ocorrencias_aeronauticas.py'])