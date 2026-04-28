import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import re
import csv
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES GERAIS E EXPERIMENTAIS
# ==========================================
PASTA_ENTRADA = "./anomaly-dataset-5/test/images" 
PASTA_LABELS_GT = "./anomaly-dataset-5/test/labels" 
PASTA_SAIDA_BASE = "./resultados_finais"

TAMANHOS_MODELOS = ['n', 's', 'm', 'l', 'x']

#ID da anomalia no dataset (conforme data.yaml)
ID_CLASSE_ANOMALIA_GT = 12 
LIMIAR_IOU_AVALIACAO = 0.30 
THRESHOLD_OCR = 0.50
TOLERANCIA_COLISAO_PX = 15 
CONFIANCA_YOLO = 0.50

CLASSES_FUNDO = ['BackgroundImage', 'Modal', 'Toolbar', 'UpperTaskBar']
CLASSES_DE_TEXTO = ['Text', 'TextButton', 'CheckedTextView', 'EditText']
EXCECOES_PERMITIDAS = [
    set(['Text', 'EditText']), set(['Text', 'CheckedTextView']),
    set(['Icon', 'EditText']), set(['Text', 'Text']),
    set(['CheckedTextView', 'CheckedTextView']), set(['Icon', 'CheckedTextView']),
]

# ==========================================
# 2. FUNÇÕES GEOMÉTRICAS E DE AVALIAÇÃO
# ==========================================
def calcular_colisao(boxA, boxB):
    xA_inter, yA_inter = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB_inter, yB_inter = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB_inter - xA_inter) * max(0, yB_inter - yA_inter)
    
    xA_union, yA_union = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
    xB_union, yB_union = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
    return interArea, [int(xA_union), int(yA_union), int(xB_union), int(yB_union)]

def eh_duplicata_do_yolo(boxA, boxB):
    wA, hA = boxA[2] - boxA[0], boxA[3] - boxA[1]
    wB, hB = boxB[2] - boxB[0], boxB[3] - boxB[1]
    cxA, cyA = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    cxB, cyB = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
    dist_x, dist_y = abs(cxA - cxB), abs(cyA - cyB)
    diff_w, diff_h = abs(wA - wB), abs(hA - hB)
    return dist_x < 8 and dist_y < 8 and diff_w < 15 and diff_h < 15

def eh_contido(boxA, boxB, threshold=0.85):
    xA_inter, yA_inter = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB_inter, yB_inter = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB_inter - xA_inter) * max(0, yB_inter - yA_inter)
    if interArea == 0: return False
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return (interArea / float(areaA)) >= threshold or (interArea / float(areaB)) >= threshold

def calcular_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

def yolo_para_xyxy(x_c, y_c, w, h, img_w, img_h):
    return [int((x_c - w/2) * img_w), int((y_c - h/2) * img_h), int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)]

# ==========================================
# 3. FUNÇÃO DE AVALIAÇÃO POR MODELO
# ==========================================
def avaliar_modelo(tamanho, leitor_ocr, arquivos):
    caminho_modelo = f"./Modelos/yolov8{tamanho}/weights/best.pt"
    pasta_saida_modelo = os.path.join(PASTA_SAIDA_BASE, f"yolov8{tamanho}")
    os.makedirs(pasta_saida_modelo, exist_ok=True)
    
    print(f"\n[{tamanho.upper()}] Processando modelo: yolov8{tamanho}")
    if not os.path.exists(caminho_modelo):
        print(f"ERRO: Modelo não encontrado em {caminho_modelo}. Pulando...")
        return None

    model = YOLO(caminho_modelo)
    
    # Contadores Nível Caixa (Bounding Box)
    TOTAL_GT_global = 0
    TOTAL_PRED_global = 0
    TP_global, FP_global, FN_global = 0, 0, 0

    # Contadores Nível Imagem (Tela inteira)
    IMG_TP, IMG_FP, IMG_FN, IMG_TN = 0, 0, 0, 0

    for arquivo in arquivos:
        caminho_img = os.path.join(PASTA_ENTRADA, arquivo)
        img = cv2.imread(caminho_img)
        img_visual = img.copy()
        altura_img, largura_img = img.shape[:2]
        
        results = model.predict(img, conf=CONFIANCA_YOLO, iou=0.85, agnostic_nms=False, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes_ids = results[0].boxes.cls.cpu().numpy()
        nomes_classes = model.names
        
        anomalias_overlap = []
        anomalias_ocr = []
        qtd_elementos = len(boxes)
        
        # Lógica OCR
        for i, box in enumerate(boxes):
            nome_classe = nomes_classes[int(classes_ids[i])]
            if nome_classe in CLASSES_DE_TEXTO:
                x1, y1, x2, y2 = map(int, box)
                pad = 5
                roi = img[max(0, y1-pad):min(altura_img, y2+pad), max(0, x1-pad):min(largura_img, x2+pad)]
                
                if roi.shape[0] > 10 and roi.shape[1] > 10:
                    res_ocr = leitor_ocr.readtext(roi)
                    if not res_ocr: continue
                    max_conf = max([c for (_, t, c) in res_ocr])
                    texto_detectado = " ".join([t for (_, t, c) in res_ocr])
                    texto_so_letras = re.sub(r'[^\w\s]', '', texto_detectado).strip()
                    if len(texto_so_letras) == 0 and "..." not in texto_detectado: continue 
                    limiar_exigido = 0.30 if "..." in texto_detectado else THRESHOLD_OCR
                    if max_conf < limiar_exigido:
                        anomalias_ocr.append({'caixa': [x1, y1, x2, y2]})

        # Lógica Overlap
        for i in range(qtd_elementos):
            nome_A = nomes_classes[int(classes_ids[i])]
            if nome_A in CLASSES_FUNDO: continue
            for j in range(i + 1, qtd_elementos):
                nome_B = nomes_classes[int(classes_ids[j])]
                if nome_B in CLASSES_FUNDO: continue
                par_atual = set([nome_A, nome_B])
                if par_atual in EXCECOES_PERMITIDAS: continue 
                if par_atual in [set(['Image', 'Text']), set(['Image', 'Icon'])] and eh_contido(boxes[i], boxes[j], 0.85): continue 
                if eh_duplicata_do_yolo(boxes[i], boxes[j]): continue 
                area_inter, caixa_combinada = calcular_colisao(boxes[i], boxes[j])
                if area_inter > TOLERANCIA_COLISAO_PX: 
                    anomalias_overlap.append({'caixa': caixa_combinada})

        # Deduplicação das predições
        todas_caixas_preditas = [a['caixa'] for a in anomalias_overlap] + [a['caixa'] for a in anomalias_ocr]
        caixas_preditas_unicas = []
        for p_box in todas_caixas_preditas:
            if not any(calcular_iou(p_box, f_box) > 0.50 for f_box in caixas_preditas_unicas):
                caixas_preditas_unicas.append(p_box)

        # Leitura do Ground Truth
        caminho_txt = os.path.join(PASTA_LABELS_GT, os.path.splitext(arquivo)[0] + ".txt")
        caixas_gt_anomalias = []
        if os.path.exists(caminho_txt):
            with open(caminho_txt, 'r') as f:
                for linha in f.readlines():
                    dados = linha.strip().split()
                    if int(dados[0]) == ID_CLASSE_ANOMALIA_GT:
                        caixas_gt_anomalias.append(yolo_para_xyxy(*map(float, dados[1:]), largura_img, altura_img))

        # Atualizando os contadores (Nível Caixa)
        TOTAL_GT_global += len(caixas_gt_anomalias)
        TOTAL_PRED_global += len(caixas_preditas_unicas)

        tp_img, fp_img = 0, 0
        gt_marcados = [False] * len(caixas_gt_anomalias)
        for pred_box in caixas_preditas_unicas:
            melhor_iou, melhor_idx = 0, -1
            for idx_gt, gt_box in enumerate(caixas_gt_anomalias):
                if not gt_marcados[idx_gt]:
                    iou = calcular_iou(pred_box, gt_box)
                    if iou > melhor_iou:
                        melhor_iou, melhor_idx = iou, idx_gt
            if melhor_iou >= LIMIAR_IOU_AVALIACAO:
                tp_img += 1
                gt_marcados[melhor_idx] = True
            else:
                fp_img += 1 

        TP_global += tp_img
        FP_global += fp_img
        FN_global += gt_marcados.count(False)

        # ====================================================
        # NOVO: AVALIAÇÃO A NÍVEL DE IMAGEM (Lógica de Negócio)
        # ====================================================
        tinha_anomalia_real = len(caixas_gt_anomalias) > 0
        modelo_acionou_alerta = len(caixas_preditas_unicas) > 0

        if tinha_anomalia_real and modelo_acionou_alerta:
            IMG_TP += 1 # Sucesso: Tela com bug foi enviada pro QA
        elif not tinha_anomalia_real and modelo_acionou_alerta:
            IMG_FP += 1 # Desperdício: QA vai olhar uma tela perfeita (Alarme falso da imagem)
        elif tinha_anomalia_real and not modelo_acionou_alerta:
            IMG_FN += 1 # Risco: Tela com bug aprovada automaticamente (Fuga)
        elif not tinha_anomalia_real and not modelo_acionou_alerta:
            IMG_TN += 1 # Sucesso Invisível: Tela boa aprovada sem gastar tempo do QA

        # Renderização Básica
        for cx1, cy1, cx2, cy2 in caixas_preditas_unicas:
            cv2.rectangle(img_visual, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(pasta_saida_modelo, arquivo), img_visual)

    # Cálculos Finais (Nível Caixa)
    precisao = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else 0
    recall = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else 0
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

    # Cálculos Finais (Nível Imagem)
    img_precisao = IMG_TP / (IMG_TP + IMG_FP) if (IMG_TP + IMG_FP) > 0 else 0
    img_recall = IMG_TP / (IMG_TP + IMG_FN) if (IMG_TP + IMG_FN) > 0 else 0
    
    return {
        'Modelo': f'yolov8{tamanho}', 
        # Métricas de Bounding Box
        'Total_GT_Box': TOTAL_GT_global, 
        'Total_Pred_Box': TOTAL_PRED_global,
        'TP_Box': TP_global, 
        'FP_Box': FP_global, 
        'FN_Box': FN_global,
        'Precisao_Box': precisao, 
        'Recall_Box': recall, 
        'F1-Score_Box': f1_score,
        # Métricas de Nível de Imagem (Telas)
        'Telas_com_Bug_Achadas (IMG_TP)': IMG_TP,
        'Telas_Boas_com_Alarme (IMG_FP)': IMG_FP,
        'Telas_com_Bug_Perdidas (IMG_FN)': IMG_FN,
        'Telas_Boas_Ignoradas (IMG_TN)': IMG_TN,
        'Precisao_Imagem': img_precisao,
        'Recall_Imagem': img_recall
    }

# ==========================================
# 4. PIPELINE DE EXECUÇÃO E PLOTAGEM
# ==========================================
def main():
    print("Iniciando Benchmarking...")
    os.makedirs(PASTA_SAIDA_BASE, exist_ok=True)
    
    print("Carregando EasyOCR UMA única vez para não perder tempo...")
    leitor_ocr = easyocr.Reader(['en', 'pt'], gpu=True) 
    
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    resultados_finais = []

    print("\n" + "="*90)
    print(f"{'MODELO':<10} | {'F1-SCORE (BOX)':<15} | {'TELAS COM ALARME FALSO':<25} | {'FUGAS DE BUGS (FN TELAS)'}")
    print("="*90)

    for tamanho in TAMANHOS_MODELOS:
        r = avaliar_modelo(tamanho, leitor_ocr, arquivos)
        if r:
            resultados_finais.append(r)
            # Imprime um resumo misturando a métrica técnica (Box F1) com a de negócio (Telas)
            print(f"yolov8{tamanho:<4} | {r['F1-Score_Box']*100:>10.1f}%     | {r['Telas_Boas_com_Alarme (IMG_FP)']:<22} | {r['Telas_com_Bug_Perdidas (IMG_FN)']}")

    if not resultados_finais:
        print("Nenhum modelo foi processado.")
        return

    # --- SALVAR EM CSV ---
    caminho_csv = os.path.join(PASTA_SAIDA_BASE, "comparativo_modelos.csv")
    with open(caminho_csv, mode='w', newline='') as file:
        fieldnames = [
            'Modelo', 'Total_GT_Box', 'Total_Pred_Box', 'TP_Box', 'FP_Box', 'FN_Box', 
            'Precisao_Box', 'Recall_Box', 'F1-Score_Box',
            'Telas_com_Bug_Achadas (IMG_TP)', 'Telas_Boas_com_Alarme (IMG_FP)', 
            'Telas_com_Bug_Perdidas (IMG_FN)', 'Telas_Boas_Ignoradas (IMG_TN)',
            'Precisao_Imagem', 'Recall_Imagem'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados_finais)
    print(f"\n✅ Tabela de dados salva em: {caminho_csv} (Verifique as novas colunas de Imagem!)")

    # --- GERAR GRÁFICO DE BARRAS (Focado nas Caixas como antes) ---
    modelos = [r['Modelo'] for r in resultados_finais]
    f1_scores = [r['F1-Score_Box'] * 100 for r in resultados_finais]
    precisoes = [r['Precisao_Box'] * 100 for r in resultados_finais]
    recalls = [r['Recall_Box'] * 100 for r in resultados_finais]

    x = np.arange(len(modelos))
    width = 0.25  

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisoes, width, label='Precisão (%)', color='#3498db')
    ax.bar(x, recalls, width, label='Recall (%)', color='#e74c3c')
    ax.bar(x + width, f1_scores, width, label='F1-Score (%)', color='#2ecc71')

    ax.set_ylabel('Porcentagem (%)')
    ax.set_title('Desempenho da Detecção de Anomalias por Modelo YOLOv8 (Nível Bounding Box)')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()
    ax.set_ylim(0, 100) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(precisoes):
        ax.text(i - width, v + 1, f"{v:.1f}%", ha='center', fontsize=8, fontweight='bold')
    for i, v in enumerate(recalls):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8, fontweight='bold')
    for i, v in enumerate(f1_scores):
        ax.text(i + width, v + 1, f"{v:.1f}%", ha='center', fontsize=8, fontweight='bold')

    caminho_grafico = os.path.join(PASTA_SAIDA_BASE, "comparativo_grafico_boxes.png")
    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=300)
    print(f"✅ Gráfico salvo em: {caminho_grafico}")
    print("\nProcesso Concluído!")

if __name__ == "__main__":
    main()