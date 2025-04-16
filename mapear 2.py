import cv2
import csv
import os
import numpy as np
from datetime import datetime

#normal:C:\Users\anjos\Downloads\WhatsApp Image 2025-03-16 at 00.26.22.jpeg
#dupla marcação: C:\Users\anjos\Downloads\WhatsApp Image 2025-03-29 at 00.30.40.jpeg

# 1. Carregar imagem
caminho_imagem = r"C:\Users\anjos\Desktop\T.I\gabritoDev ideias\GABARITO RECORTADO\gabarito_recortado.jpg"
imagem = cv2.imread(caminho_imagem)
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# 2. Pré-processamento
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
_, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Remoção de ruído
kernel = np.ones((5, 5), np.uint8)
mascara = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

# 4. Redimensionamento
largura_desejada = 678
escala = largura_desejada / mascara.shape[1]
altura_desejada = int(mascara.shape[0] * escala)
mascara_redimensionada = cv2.resize(mascara, (largura_desejada, altura_desejada))

# 5. Detectar bolhas
contornos, _ = cv2.findContours(mascara_redimensionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. Filtrar bolhas válidas
area_minima = 200
limiar_branco = 0.75
bolhas_validas = []

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > area_minima:
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * area / (perimetro ** 2)
        if circularidade > 0.4:
            mascara_local = np.zeros_like(mascara_redimensionada)
            cv2.drawContours(mascara_local, [contorno], -1, 255, -1)
            pixels_brancos = np.sum(mascara_redimensionada[mascara_local == 255] == 255)
            total_pixels = np.sum(mascara_local == 255)
            if total_pixels > 0:
                proporcao_branco = pixels_brancos / total_pixels
                if proporcao_branco >= limiar_branco:
                    bolhas_validas.append(contorno)

# 7. Ordenar bolhas por posição vertical (Y)
bolhas_validas = sorted(bolhas_validas, key=lambda c: cv2.boundingRect(c)[1])

# 8. Extrair centróides
centroides = []
for i, contorno in enumerate(bolhas_validas[:180]):  # 3 blocos de 60 bolhas
    M = cv2.moments(contorno)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroides.append((cx, cy, contorno, i+1))  # i+1 é o número da marcação

# 9. Definir sistema de classificação para cada bloco
num_questoes_por_bloco = 20
num_blocos = 3
letras_colunas = ['A', 'B', 'C', 'D', 'E']

# Posições X das colunas para cada bloco
posicoes_colunas_por_bloco = {
    0: {'A': 75, 'B': 104, 'C': 134, 'D': 163, 'E': 192},    # Bloco 1-20
    1: {'A': 301, 'B': 330, 'C': 357, 'D': 387, 'E': 415},   # Bloco 21-40
    2: {'A': 525, 'B': 553, 'C': 584, 'D': 612, 'E': 641}    # Bloco 41-60
}

# Limites horizontais aproximados para cada bloco
limites_x_blocos = {
    0: (0, 250),    # Bloco 1
    1: (250, 450),  # Bloco 2
    2: (450, 700)   # Bloco 3
}

margem_coluna = 15

# 10. Classificar bolhas para todos os blocos com verificação de dupla marcação
gabarito_completo = {}
for bloco in range(num_blocos):
    gabarito_bloco = {q: {'respostas': {letra: None for letra in letras_colunas}, 'valida': True} 
                     for q in range(1, num_questoes_por_bloco+1)}
    gabarito_completo[bloco] = gabarito_bloco

# Ordenar centróides por posição Y (linha)
centroides.sort(key=lambda x: x[1])

# Processar cada bloco separadamente
for bloco in range(num_blocos):
    # Filtrar centróides que pertencem a este bloco
    centroides_bloco = [(x, y, cnt, num) for (x, y, cnt, num) in centroides 
                       if limites_x_blocos[bloco][0] <= x < limites_x_blocos[bloco][1]]
    
    if not centroides_bloco:
        continue
        
    # Distribuir bolhas igualmente entre as questões neste bloco
    passo_y = (centroides_bloco[-1][1] - centroides_bloco[0][1]) / (num_questoes_por_bloco - 1)
    
    for x, y, cnt, num_marcacao in centroides_bloco:
        questao = min(int(round((y - centroides_bloco[0][1]) / passo_y)) + 1, num_questoes_por_bloco)
        
        # Classificar por coluna usando as posições específicas do bloco
        distancias = {letra: abs(x - pos) for letra, pos in posicoes_colunas_por_bloco[bloco].items()}
        coluna = min(distancias, key=distancias.get)
        
        if distancias[coluna] <= margem_coluna:
            gabarito_completo[bloco][questao]['respostas'][coluna] = (num_marcacao, x, y, cnt)

# Verificar dupla marcação e marcar questões inválidas
for bloco in range(num_blocos):
    for questao in range(1, num_questoes_por_bloco+1):
        respostas = [letra for letra in letras_colunas 
                    if gabarito_completo[bloco][questao]['respostas'][letra] is not None]
        
        if len(respostas) > 1:
            gabarito_completo[bloco][questao]['valida'] = False
            questao_real = questao + (bloco * num_questoes_por_bloco)
            print(f"ATENÇÃO: Questão {questao_real} tem múltiplas marcações ({', '.join(respostas)}) - Considerada inválida")

# 11. Gerar imagem de saída com marcações coloridas
saida = cv2.cvtColor(mascara_redimensionada, cv2.COLOR_GRAY2BGR)
cores = {'A': (255,0,0), 'B': (0,255,0), 'C': (0,0,255), 'D': (255,255,0), 'E': (255,0,255)}

for bloco in range(num_blocos):
    for questao in range(1, num_questoes_por_bloco+1):
        valida = gabarito_completo[bloco][questao]['valida']
        for letra in letras_colunas:
            if gabarito_completo[bloco][questao]['respostas'][letra]:
                num_marcacao, x, y, cnt = gabarito_completo[bloco][questao]['respostas'][letra]
                raio = int(cv2.minEnclosingCircle(cnt)[1])
                cor = (0, 0, 255) if not valida else cores[letra]  # Vermelho para inválidas
                cv2.circle(saida, (x, y), raio, cor, 2)
                questao_real = questao + (bloco * num_questoes_por_bloco)
                cv2.putText(saida, f"{questao_real}{letra}", (x+15, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)

# 12. Mostrar resultados
cv2.imshow("Gabarito Classificado (Todos os blocos)", saida)

# 13. Gerar relatório
print(f"\nRELATÓRIO DE PROCESSAMENTO")
print(f"Total de bolhas detectadas: {len(centroides)}")
print(f"Total de questões com dupla marcação: {sum(1 for bloco in range(num_blocos) for q in range(1, num_questoes_por_bloco+1) if not gabarito_completo[bloco][q]['valida'])}")

# 14. Gerar arquivo CSV apenas com respostas válidas
def gerar_csv(gabarito_completo, num_blocos, num_questoes_por_bloco, letras_colunas):
    dados = []
    respostas_invalidas = 0
    
    for bloco in range(num_blocos):
        for questao in range(1, num_questoes_por_bloco + 1):
            if not gabarito_completo[bloco][questao]['valida']:
                respostas_invalidas += 1
                continue
                
            for letra in letras_colunas:
                if gabarito_completo[bloco][questao]['respostas'][letra]:
                    questao_real = questao + (bloco * num_questoes_por_bloco)
                    dados.append({
                        "Questão": str(questao_real),
                        "Resposta": letra
                    })
    
    diretorio = r"C:\Users\anjos\Desktop\T.I\gabritoDev ideias\RESPOSTAS EXTRAIDAS"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"GABARITO_EXTRAIDO_{timestamp}.csv"
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    
    os.makedirs(diretorio, exist_ok=True)
    
    with open(caminho_completo, mode='w', newline='', encoding='utf-8') as arquivo:
        campos = ["Questão", "Resposta"]
        writer = csv.DictWriter(arquivo, fieldnames=campos, delimiter=';')
        
        writer.writeheader()
        writer.writerows(dados)
    
    print(f"\nArquivo CSV gerado com sucesso em: {caminho_completo}")
    print(f"Total de respostas válidas exportadas: {len(dados)}")
    print(f"Total de respostas inválidas (dupla marcação): {respostas_invalidas}")

# Chamar a função para gerar o CSV
gerar_csv(gabarito_completo, num_blocos, num_questoes_por_bloco, letras_colunas)

cv2.waitKey(0)
cv2.destroyAllWindows()