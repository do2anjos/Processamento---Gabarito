import cv2
import numpy as np

def cortar_gabarito_preciso(caminho_imagem, caminho_saida=None):
    """Corta o gabarito com margens de segurança e ajuste preciso"""
    # 1. Carregar imagem
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro ao carregar a imagem")
        return None
    
    # 2. Processamento para destacar a estrutura do gabarito
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    
    # 3. Dilatar as bordas para conectar linhas próximas
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    # 4. Encontrar contornos do gabarito
    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print("Não foi possível identificar o gabarito")
        return None
    
    # 5. Pegar o maior contorno retangular
    maior_contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_contorno)
    
    # 6. Adicionar margens de segurança (ajuste esses valores conforme necessário)
    margem_superior = 580 # Margem extra no topo
    margem_lateral = 1   # Margem nas laterais
    
    x = max(0, x - margem_lateral)
    y = max(0, y - margem_superior)
    w = min(img.shape[1] - x, w + 2 * margem_lateral)
    h = min(img.shape[0] - y, h + margem_superior + 10)  # +10 na base
    
    # 7. Ajustar proporção mantendo as margens
    proporcao_alvo = 1100 / 810
    proporcao_atual = w / h
    
    if proporcao_atual > proporcao_alvo:
        # Ajustar altura mantendo a margem superior
        nova_altura = int(w / proporcao_alvo)
        h = nova_altura
    else:
        # Ajustar largura mantendo as margens laterais
        nova_largura = int(h * proporcao_alvo)
        x = max(0, x - (nova_largura - w) // 2)
        w = nova_largura
    
    # 8. Recortar a região com as margens
    gabarito = img[y:y+h, x:x+w]
    
    # 9. Redimensionar para o tamanho desejado
    gabarito = cv2.resize(gabarito, (680, 525))
    
    # 10. Salvar a imagem se um caminho de saída foi fornecido
    if caminho_saida:
        try:
            cv2.imwrite(caminho_saida, gabarito, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"Imagem salva com sucesso em: {caminho_saida}")
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")
    
    # 11. Exibir resultado
    cv2.imshow("Gabarito Recortado com Margens", gabarito)
    print("Pressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return gabarito

if __name__ == "__main__":
    caminho_entrada = r"C:\Users\anjos\Downloads\WhatsApp Image 2025-03-28 at 12.23.17.jpeg"
    caminho_saida = r"C:\Users\anjos\Desktop\T.I\gabritoDev ideias\GABARITO RECORTADO\gabarito_recortado.jpg"
    
    gabarito = cortar_gabarito_preciso(caminho_entrada, caminho_saida)  