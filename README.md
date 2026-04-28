Conteúdo do Repositório

1. Conjuntos de Dados

Dataset VINS: Recorte de interfaces íntegras utilizado para o treinamento e validação do modelo de detecção de objetos (YOLOv8).

Dataset de Avaliação: Base de 453 imagens para teste do sistema completo:

197 imagens com anomalias induzidas manualmente (sobreposições e legibilidade).

256 imagens de interfaces íntegras para validação de falsos positivos.

2. Algoritmo e Código-Fonte

Módulo de Detecção: Implementação da inferência YOLOv8 para localização de componentes.

Algoritmo de Regras: Pós-processamento heurístico que aplica:

Remoção de duplicatas (proximidade < 8px) e filtros de classes de fundo.

Cálculo de colisões geométricas (limiar de intersecção > 15px).

Regra de contenção (limiar de 85% da área).

Análise de legibilidade via OCR (limiares de confiança de 0,50 e 0,30).

Aplicação de exceções para pares de componentes legítimos.
