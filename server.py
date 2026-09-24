from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware

import cv2
import io
import os
import requests
import numpy as np

from PIL import Image
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURAÇÃO DA API DO KAIRÓS
# ============================================================


URL_BIOMETRIAS_KAIROS = "https://sitekairos.onrender.com/api/cameras/biometrias"
URL_AUTENTICACAO_FACIAL_KAIROS = (
    "https://sitekairos.onrender.com/api/cameras/autenticacao-facial"
)
CACHE_BIOMETRIAS = []
CACHE_BIOMETRIAS_CARREGADO = False

def buscar_biometrias_kairos():
    chave_api = os.getenv("CAMERA_API_KEY")

    if not chave_api:
        raise RuntimeError(
            "CAMERA_API_KEY não está configurada."
        )

    resposta = requests.post(
    URL_AUTENTICACAO_FACIAL_KAIROS,
    headers={
            "x-camera-api-key": chave_api
        },
        timeout=15
    )

    resposta.raise_for_status()

    dados = resposta.json()

    return dados.get("usuarios", [])

def enviar_embedding_para_kairos(embedding):
    chave_api = os.getenv("CAMERA_API_KEY")

    if not chave_api:
        raise RuntimeError(
            "CAMERA_API_KEY não está configurada."
        )

    resposta = requests.post(
        URL_AUTENTICACAO_FACIAL_KAIROS,
        headers={
            "x-camera-api-key": chave_api,
            "Content-Type": "application/json"
        },
        json={
            "biometria": embedding
        },
        timeout=15
    )

    resposta.raise_for_status()

    return resposta.json()
# ============================================================
# BANCO FACIAL LOCAL
# ============================================================

ARQUIVO_BANCO = "banco_faces.npy"

# Carrega as faces salvas do disco ao iniciar o servidor
if os.path.exists(ARQUIVO_BANCO):
    BANCO_FACIAL = np.load(
        ARQUIVO_BANCO,
        allow_pickle=True
    ).item()
else:
    BANCO_FACIAL = {}


def salvar_banco():
    """Salva os embeddings no arquivo físico .npy"""
    np.save(ARQUIVO_BANCO, BANCO_FACIAL)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# Configuração de CORS para permitir acesso do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INSIGHTFACE / ARCFACE
# ============================================================

app_face = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app_face.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# ROTA TEMPORÁRIA PARA TESTAR AS BIOMETRIAS
# ============================================================

@app.get("/teste-biometrias")
def teste_biometrias():
    try:
        global CACHE_BIOMETRIAS
        global CACHE_BIOMETRIAS_CARREGADO

        if not CACHE_BIOMETRIAS_CARREGADO:
            print("BUSCANDO BIOMETRIAS DO KAIRÓS...")
            CACHE_BIOMETRIAS = buscar_biometrias_kairos()
            CACHE_BIOMETRIAS_CARREGADO = True
            print(f"{len(CACHE_BIOMETRIAS)} BIOMETRIAS CARREGADAS.")

        usuarios = CACHE_BIOMETRIAS

        return {
            "sucesso": True,
            "quantidade": len(usuarios),
            "usuarios": [
                {
                    "id": usuario.get("id"),
                    "nome": usuario.get("nome"),
                    "dimensoes": len(usuario.get("biometria", []))
                }
                for usuario in usuarios
            ]
        }

    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }


# ============================================================
# CADASTRAR ROSTO
# ============================================================

@app.post("/cadastrar-rosto")
async def cadastrar_rosto(
    nome: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Rota para cadastrar o rosto de um operador
    no banco de dados local.
    """

    try:
        conteudo = await file.read()

        imagem = Image.open(
            io.BytesIO(conteudo)
        ).convert("RGB")

        img_np = np.array(imagem)

        faces = app_face.get(img_np)

        if not faces:
            return {
                "sucesso": False,
                "erro": "Nenhum rosto detectado na imagem."
            }

        # Armazena o vetor de características
        # da primeira face detectada
        BANCO_FACIAL[nome] = faces[0].embedding

        # Salva a alteração no arquivo banco_faces.npy
        salvar_banco()

        return {
            "sucesso": True,
            "mensagem": f"Rosto de '{nome}' cadastrado e salvo com sucesso!"
        }

    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }


# ============================================================
# ANALISAR IMAGEM
# ============================================================

@app.post("/analisar")
async def analisar_imagem(
    file: UploadFile = File(...)
):
    """
    Rota chamada pelo KAIRÓS para reconhecer
    a pessoa capturada na câmera.
    """

    try:
        conteudo = await file.read()

        imagem = Image.open(
            io.BytesIO(conteudo)
        ).convert("RGB")

        img_np = np.array(imagem)

        faces = app_face.get(img_np)

        deteccoes = []

        for face in faces:

            embedding_atual = face.embedding

            melhor_nome = "Desconhecido"
            maior_similaridade = 0.0

            # Comparação vetorial via Cosseno
            # contra as pessoas cadastradas
            for nome_cadastrado, emb_cadastrado in BANCO_FACIAL.items():

                sim = np.dot(
                    embedding_atual,
                    emb_cadastrado
                ) / (
                    np.linalg.norm(embedding_atual)
                    * np.linalg.norm(emb_cadastrado)
                )

                if sim > maior_similaridade:
                    maior_similaridade = sim
                    melhor_nome = nome_cadastrado

            # Converte similaridade para porcentagem
            confianca_percentual = round(
                float(maior_similaridade) * 100,
                2
            )

            # Limiar mínimo de similaridade
            if maior_similaridade < 0.40:
                melhor_nome = "Desconhecido"

            deteccoes.append({
                "classe": melhor_nome,
                "confianca": confianca_percentual,
                "bbox": [
                    int(b)
                    for b in face.bbox
                ]
            })

        return {
            "sucesso": True,
            "deteccoes": deteccoes
        }

    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }
@app.post("/teste-autenticacao-sofia")
async def teste_autenticacao_sofia(
    file: UploadFile = File(...)
):
    try:
        # Recebe a imagem enviada
        conteudo = await file.read()

        imagem = Image.open(
            io.BytesIO(conteudo)
        ).convert("RGB")

        img_np = np.array(imagem)

        # Detecta o rosto e gera o embedding
        faces = app_face.get(img_np)

        if not faces:
            return {
                "sucesso": False,
                "erro": "Nenhum rosto detectado na imagem."
            }

        # Primeiro rosto encontrado
        embedding_real = faces[0].embedding

        # Normaliza o vetor
        embedding_real = (
            embedding_real /
            np.linalg.norm(embedding_real)
        )

        # Converte para lista para enviar em JSON
        embedding_lista = (
            embedding_real
            .astype(float)
            .tolist()
        )

        print(
            "Embedding gerado:",
            len(embedding_lista),
            "dimensões"
        )

        # Envia para a API da Sofia
        resultado = enviar_embedding_para_kairos(
            embedding_lista
        )

        return {
            "sucesso": True,
            "embedding_dimensoes": len(embedding_lista),
            "resultado_sofia": resultado
        }

    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }

# ============================================================
# RECONHECIMENTO FACIAL KAIRÓS - SFACE 128D
# ============================================================

@app.post("/reconhecer")
def reconhecer_rosto(dados: dict = Body(...)):
    try:
        embedding_recebido = dados.get("embedding")

        if not isinstance(embedding_recebido, list):
            return {
                "sucesso": False,
                "erro": "Embedding inválido."
            }

        if len(embedding_recebido) != 128:
            return {
                "sucesso": False,
                "erro": "O embedding deve possuir 128 dimensões."
            }

        embedding_atual = np.array(
            embedding_recebido,
            dtype=np.float32
        )

        norma_atual = np.linalg.norm(embedding_atual)

        if norma_atual == 0:
            return {
                "sucesso": False,
                "erro": "Embedding inválido."
            }

        embedding_atual = embedding_atual / norma_atual

        usuarios = buscar_biometrias_kairos()

        melhor_usuario = None
        maior_similaridade = -1.0

        for usuario in usuarios:

            biometria = usuario.get("biometria", [])

            if not isinstance(biometria, list):
                continue

            if len(biometria) != 128:
                continue

            embedding_banco = np.array(
                biometria,
                dtype=np.float32
            )

            norma_banco = np.linalg.norm(embedding_banco)

            if norma_banco == 0:
                continue

            embedding_banco = (
                embedding_banco / norma_banco
            )

            similaridade = float(
                np.dot(
                    embedding_atual,
                    embedding_banco
                )
            )

            if similaridade > maior_similaridade:
                maior_similaridade = similaridade
                melhor_usuario = usuario

        # Limiar inicial.
        # Depois calibraremos com testes reais das câmeras.
        LIMIAR_RECONHECIMENTO = 0.55

        reconhecido = (
            melhor_usuario is not None
            and maior_similaridade >= LIMIAR_RECONHECIMENTO
        )

        return {
            "sucesso": True,
            "reconhecido": reconhecido,
            "usuario": {
                "id": melhor_usuario.get("id"),
                "nome": melhor_usuario.get("nome")
            } if reconhecido else None,
            "similaridade": round(
                maior_similaridade,
                4
            )
        }

    except Exception as e:
        print(
            "ERRO NO RECONHECIMENTO FACIAL:",
            e
        )

        return {
            "sucesso": False,
            "erro": str(e)
        }


# ============================================================
# INICIAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )