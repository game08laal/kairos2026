from fastapi import FastAPI, UploadFile, File, Form, Body, Request
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
        print("AVISO: CAMERA_API_KEY não está configurada nas variáveis de ambiente.")
        return []

    try:
        headers = {"x-camera-api-key": chave_api}

        # Tenta buscar as biometrias no endpoint principal
        resposta = requests.get(
            URL_BIOMETRIAS_KAIROS,
            headers=headers,
            timeout=10
        )

        if not resposta.ok:
            # Fallback para o endpoint secundário se necessário
            resposta = requests.post(
                URL_AUTENTICACAO_FACIAL_KAIROS,
                headers=headers,
                timeout=10
            )

        resposta.raise_for_status()
        dados = resposta.json()

        if isinstance(dados, list):
            return dados
        return dados.get("usuarios", dados.get("biometrias", []))

    except Exception as e:
        print(f"ERRO AO BUSCAR BIOMETRIAS DO KAIRÓS: {e}")
        return []


def enviar_embedding_para_kairos(embedding):
    chave_api = os.getenv("CAMERA_API_KEY")

    if not chave_api:
        raise RuntimeError("CAMERA_API_KEY não está configurada.")

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

app = FastAPI(
    title="Servidor de Reconhecimento Facial Kairós",
    version="1.0"
)


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

        return {
            "sucesso": True,
            "total_carregados": len(CACHE_BIOMETRIAS),
            "biometrias": CACHE_BIOMETRIAS
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

        # Armazena o vetor de características da primeira face
        BANCO_FACIAL[nome] = faces[0].embedding

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

            confianca_percentual = round(
                float(maior_similaridade) * 100,
                2
            )

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

        embedding_real = faces[0].embedding

        embedding_real = (
            embedding_real /
            np.linalg.norm(embedding_real)
        )

        embedding_lista = (
            embedding_real
            .astype(float)
            .tolist()
        )

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
def reconhecer_farmaceutico(dados: dict):
    """
    Recebe o embedding SFace 128D do front-end
    e envia para o backend oficial do KAIRÓS.
    """

    try:
        embedding_recebido = dados.get("embedding", [])

        if not isinstance(embedding_recebido, list):
            return {
                "sucesso": False,
                "reconhecido": False,
                "erro": "Embedding inválido."
            }

        if len(embedding_recebido) != 128:
            return {
                "sucesso": False,
                "reconhecido": False,
                "erro": (
                    f"Embedding inválido: "
                    f"{len(embedding_recebido)} dimensões. "
                    "Esperado: 128."
                )
            }

        print(
            "[RECONHECIMENTO] Enviando embedding "
            "128D para o KAIRÓS..."
        )

        resposta_sofia = enviar_embedding_para_kairos(
            embedding_recebido
        )

        print(
            "[RECONHECIMENTO] Resposta recebida "
            "do KAIRÓS."
        )

        if (
            resposta_sofia.get("autenticado")
            or resposta_sofia.get("reconhecido")
        ):
            usuario_info = resposta_sofia.get(
                "usuario",
                {}
            )

            token = resposta_sofia.get("token")

            print(
                "[RECONHECIMENTO] Usuário autenticado:",
                usuario_info.get("id"),
                usuario_info.get("nome")
            )

            print(
                "[RECONHECIMENTO] JWT recebido:",
                "SIM" if token else "NÃO"
            )

            return {
                "sucesso": True,
                "reconhecido": True,
                "usuario": {
                    "id": usuario_info.get("id"),
                    "nome": usuario_info.get("nome"),
                    "email": usuario_info.get("email"),
                    "cargo": usuario_info.get("cargo")
                },
                "similaridade": resposta_sofia.get(
                    "similaridade"
                ),
                "maiorSimilaridade": resposta_sofia.get(
                    "similaridade"
                ),
                "token": token
            }

        print(
            "[RECONHECIMENTO] Rosto não reconhecido "
            "pelo KAIRÓS."
        )

        return {
            "sucesso": True,
            "reconhecido": False,
            "usuario": None,
            "similaridade": resposta_sofia.get(
                "similaridade"
            ),
            "maiorSimilaridade": resposta_sofia.get(
                "similaridade"
            ),
            "token": None,
            "erro": resposta_sofia.get(
                "erro",
                "Rosto não reconhecido."
            )
        }

    except requests.HTTPError as e:
        resposta = e.response

        if resposta is not None:
            try:
                dados_erro = resposta.json()
            except Exception:
                dados_erro = {
                    "erro": resposta.text
                }

            print(
                "[RECONHECIMENTO] KAIRÓS respondeu:",
                resposta.status_code,
                dados_erro
            )

            return {
                "sucesso": True,
                "reconhecido": False,
                "usuario": None,
                "token": None,
                "erro": dados_erro.get(
                    "erro",
                    "Rosto não reconhecido."
                ),
                "similaridade": dados_erro.get(
                    "similaridade"
                )
            }

        return {
            "sucesso": False,
            "reconhecido": False,
            "usuario": None,
            "token": None,
            "erro": str(e)
        }

    except Exception as e:
        print(
            "[ERRO RECONHECER]",
            e
        )

        return {
            "sucesso": False,
            "reconhecido": False,
            "usuario": None,
            "token": None,
            "erro": str(e)
        }

# ============================================================
# ROTA DE PACIENTES DO KAIRÓS
# ============================================================

@app.get("/pacientes")
def obter_pacientes(
    usuario_id: str = "",
    request: Request = None
):
    """
    Consulta os pacientes no backend central do Render.
    """
    try:
        url = "https://sitekairos.onrender.com/api/pacientes"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Captura o token enviado pelo front-end (pacientes.js)
        authorization = request.headers.get("Authorization", "") if request else ""
        
        if authorization:
            headers["Authorization"] = authorization

        print(f"[SERVER] Requisitando pacientes no Render | ID Operador: '{usuario_id}'")

        resposta = requests.get(
            url,
            params={"usuario_id": usuario_id} if usuario_id else {},
            headers=headers,
            timeout=10
        )

        print(f"[SERVER] Status HTTP Central: {resposta.status_code}")

        if resposta.ok:
            dados = resposta.json()
            pacientes = dados if isinstance(dados, list) else dados.get("pacientes", [])
            print(f"[SERVER] Pacientes retornados: {len(pacientes)}")

            if not usuario_id:
                return pacientes

            # Filtro local de segurança
            filtrados = [
                p for p in pacientes
                if str(p.get("usuario_id") or p.get("usuarioId") or p.get("user_id") or "") == str(usuario_id)
            ]

            return filtrados if len(filtrados) > 0 else pacientes

        print(f"[SERVER] Erro na API Central ({resposta.status_code}): {resposta.text}")
        return []

    except Exception as e:
        print(f"[SERVER] Exceção ao buscar pacientes: {e}")
        return []


# ============================================================
# INICIAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("===== INICIANDO SERVIDOR FASTAPI KAIRÓS (PORTA 8000) =====")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )