from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

ARQUIVO_BANCO = "banco_faces.npy"

# Carrega as faces salvas do disco ao iniciar o servidor
if os.path.exists(ARQUIVO_BANCO):
    BANCO_FACIAL = np.load(ARQUIVO_BANCO, allow_pickle=True).item()
else:
    BANCO_FACIAL = {}

def salvar_banco():
    """Salva os embeddings no arquivo físico .npy"""
    np.save(ARQUIVO_BANCO, BANCO_FACIAL)

app = FastAPI()

# Configuração de CORS para permitir acesso do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o InsightFace / ArcFace
app_face = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))


@app.post("/cadastrar-rosto")
async def cadastrar_rosto(nome: str = Form(...), file: UploadFile = File(...)):
    """
    Rota para cadastrar o rosto de um operador no banco de dados.
    """
    try:
        conteudo = await file.read()
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
        img_np = np.array(imagem)
        
        faces = app_face.get(img_np)
        if not faces:
            return {"sucesso": False, "erro": "Nenhum rosto detectado na imagem."}
        
        # Armazena o vetor de características da primeira face detectada
        BANCO_FACIAL[nome] = faces[0].embedding
        
        # Salva a alteração no arquivo banco_faces.npy
        salvar_banco()
        
        return {"sucesso": True, "mensagem": f"Rosto de '{nome}' cadastrado e salvo com sucesso!"}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}


@app.post("/analisar")
async def analisar_imagem(file: UploadFile = File(...)):
    """
    Rota chamada pelo KAIRÓS para reconhecer a pessoa capturada na câmera.
    """
    try:
        conteudo = await file.read()
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
        img_np = np.array(imagem)
        
        faces = app_face.get(img_np)
        
        deteccoes = []
        for face in faces:
            embedding_atual = face.embedding
            melhor_nome = "Desconhecido"
            maior_similaridade = 0.0
            
            # Comparação vetorial via Cosseno contra as pessoas cadastradas
            for nome_cadastrado, emb_cadastrado in BANCO_FACIAL.items():
                sim = np.dot(embedding_atual, emb_cadastrado) / (
                    np.linalg.norm(embedding_atual) * np.linalg.norm(emb_cadastrado)
                )
                if sim > maior_similaridade:
                    maior_similaridade = sim
                    melhor_nome = nome_cadastrado
            
            # Limiar mínimo de similaridade para considerar reconhecido (40%)
            confianca_percentual = round(float(maior_similaridade) * 100, 2)
            if maior_similaridade < 0.40:
                melhor_nome = "Desconhecido"

            deteccoes.append({
                "classe": melhor_nome,
                "confianca": confianca_percentual,
                "bbox": [int(b) for b in face.bbox]
            })
        
        return {"sucesso": True, "deteccoes": deteccoes}

    except Exception as e:
        return {"sucesso": False, "erro": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)