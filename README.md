# Detecção Híbrida de Anomalias em GUIs Móveis

## ⚠️ Aviso aos Revisores: Acesso aos Dados

Devido aos limites de tamanho de arquivo (Git LFS) da plataforma de anonimização, as imagens e arquivos `.zip` não puderam ser espelhados nativamente neste repositório. Para garantir a total reprodutibilidade da pesquisa, os conjuntos de dados completos estão hospedados de forma segura e independente.

📥 **[CLIQUE AQUI PARA BAIXAR O DATASET COMPLETO](https://zenodo.org/records/20358336?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ5MTE4OTU1LWI1OGMtNGY5MS1iNDcwLTAwMjJhYjFlN2M3NSIsImRhdGEiOnt9LCJyYW5kb20iOiJmZjhiYzIwNWE1NGQwNWQwZjRhNGQ5NmJlZTg1NmJmOSJ9.z-Ilm9E6MdKfaQW3Zn9-pzSxLC0vXTC8qa-5NTPK5mb4FzJXdcmSBx_QmsWB40NVXk4i3XFjPLy9fZRQP6_lMg)**

_Nota: O link acima foi gerado para acesso anônimo, preservando integralmente o processo de revisão duplo-cega (Double-Blind Review)._

---

## Conteúdo do Repositório

### 1. Algoritmo e Código-Fonte

- **Módulo de Detecção:** Implementação da inferência do YOLOv8 para a localização e extração espacial dos componentes da interface.
- **Algoritmo de Regras:** _Pipeline_ de pós-processamento que recebe as _bounding boxes_ e identifica falhas estruturais aplicando:
  - Filtros de ruído (remoção de duplicatas) e de classes de fundo.
  - Validação de colisões geométricas e regras de contenção.
  - Análise de legibilidade integrada ao EasyOCR.
  - Matriz de exceções para pares de componentes com sobreposição legítima de _design_.

### 2. Conjuntos de Dados (Disponíveis via link externo)

- **Dataset VINS:** Recortes de interfaces íntegras utilizados exclusivamente para o treinamento e validação espacial do YOLOv8.
- **Dataset de Avaliação (453 imagens):** Base construída para o teste do sistema híbrido completo, composta por:
  - 197 imagens com anomalias de _layout_ induzidas sinteticamente (sobreposições e falhas de legibilidade).
  - 256 imagens de interfaces íntegras, utilizadas como grupo de controle para medição de falsos positivos.
