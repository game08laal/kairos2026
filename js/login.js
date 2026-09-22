// =====================================================
// LOGIN KAIRÓS - Central de Monitoramento
// =====================================================

const API_LOGIN_URL = "https://sitekairos.onrender.com/api/login";

const formLogin = document.getElementById("form-login");
const erroLogin = document.getElementById("erro-login");
const botaoEntrar = document.getElementById("btn-entrar");

formLogin.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("email").value.trim();
    const senha = document.getElementById("senha").value;

    erroLogin.innerText = "";

    if (!email || !senha) {
        erroLogin.innerText = "Preencha o email e a senha.";
        return;
    }

    botaoEntrar.disabled = true;
    botaoEntrar.innerText = "Entrando...";

    try {
        const resposta = await fetch(API_LOGIN_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ email, senha })
        });

        const dados = await resposta.json();

        if (!resposta.ok) {
            erroLogin.innerText = dados.erro || "Email ou senha incorretos.";
            botaoEntrar.disabled = false;
            botaoEntrar.innerText = "Entrar";
            return;
        }

        // Login OK: guarda token e dados do usuário para o kairos.html usar
        sessionStorage.setItem("kairos_token", dados.token);
        sessionStorage.setItem("kairos_usuario", JSON.stringify(dados.usuario));

        // Redireciona para o painel de câmeras
        window.location.href = "kairos.html";

    } catch (erro) {
        console.error("Erro ao fazer login:", erro);
        erroLogin.innerText = "Não foi possível conectar ao servidor. O sistema pode estar iniciando (pode levar até 1 minuto na primeira tentativa) — tente novamente em instantes.";
        botaoEntrar.disabled = false;
        botaoEntrar.innerText = "Entrar";
    }
});