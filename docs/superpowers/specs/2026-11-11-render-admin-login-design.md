# Login do Admin no Render

## Objetivo

Permitir abrir o Admin pelo link público do Render no navegador do celular, autenticar sem colocar segredo na URL e então configurar provedores e modelos.

## Abordagens

- Bearer manual por cabeçalho: seguro para APIs, mas navegação normal do celular não consegue enviar o cabeçalho ao abrir `/admin`.
- Token na URL: simples, mas expõe a credencial em histórico, logs e compartilhamentos.
- **Cookie de sessão HttpOnly, Secure e SameSite Strict: escolhido**, porque funciona em navegação normal e mantém o token fora da URL e do JavaScript persistente.

## Decisões

- `/admin` sem sessão exibe uma tela de login móvel que pede o valor de `FCC_ADMIN_AUTH_TOKEN`.
- Login válido cria cookie com prazo de 12 horas; logout remove o cookie.
- O guard do Admin aceita sessão válida ou o Bearer já usado pelo app Android. A exceção loopback atual permanece.
- O token de login nunca será enviado na query string nem gravado em localStorage.
- OpenAI será configurado no Admin após login; nenhuma chave OpenAI será inventada ou embutida no APK.

## Fora de escopo

- Contas de usuário, OAuth e recuperação de senha.
- Mudar o provedor/modelo ativo sem uma credencial OpenAI válida.
