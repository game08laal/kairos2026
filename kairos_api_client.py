import httpx

KAIROS_BASE_URL = "https://sitekairos.onrender.com"


async def autenticar_operador_por_biometria(vetor_128d: list[float]) -> dict:
    """Envia o vetor facial de 128 dimensões para o sitekairos."""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{KAIROS_BASE_URL}/api/cameras/autenticacao-facial",
            json={"biometria": vetor_128d},
            timeout=15.0
        )

        response.raise_for_status()

        return response.json()


async def buscar_pacientes(token_jwt: str, usuario_id: str) -> dict:
    """Busca os pacientes do operador no sitekairos."""

    headers = {
        "Authorization": f"Bearer {token_jwt}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{KAIROS_BASE_URL}/api/pacientes",
            params={"usuario_id": usuario_id},
            headers=headers,
            timeout=15.0
        )

        response.raise_for_status()

        return response.json()