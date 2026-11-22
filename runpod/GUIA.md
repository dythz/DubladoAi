# LuminaDub AI no RunPod

Dubla os vídeos numa GPU na nuvem enquanto a sua placa fica livre pra trabalhar.
O programa detecta a placa do pod sozinho e se ajusta (veja o README).

## 1. Criar o pod (uma vez)

1. **Pods → Deploy**, escolha a GPU:
   - **L40S** — a mais rápida pelo preço (~4–5 min por vídeo de 15 min)
   - **RTX 4090** — tão rápida quanto, mais barata quando tem
   - **RTX A6000** — mais barata por hora, ~7 min por vídeo
2. Template: **RunPod PyTorch** (qualquer versão — o script instala o PyTorch certo).
3. **Add volume → Network Volume** de **60 GB** montado em `/workspace`.
   Ambiente Python, modelos e vídeos ficam nele: religar o pod não baixa nada de novo.
4. Em **Expose HTTP Ports**, coloque **`5000`** (além do 8888 que já vem).
   Com 2 placas no mesmo pod, coloque **`5000,5001`** (veja "Duas placas" abaixo).
5. **Deploy**.

## 2. Instalar e ligar

No pod, abra **Connect → Web Terminal** (ou o Jupyter → Terminal) e rode:

```bash
cd /workspace
git clone https://github.com/SEU_USUARIO/SEU_REPO.git LuminaDub
bash LuminaDub/runpod/iniciar.sh
```

> Repositório privado? Use `https://SEU_USUARIO:SEU_TOKEN@github.com/...` no clone
> (token em GitHub → Settings → Developer settings → Personal access tokens).

A primeira vez leva uns **10–15 min** (instala tudo e baixa ~8 GB de modelos).
No final aparece o link, algo como:

```
Abra no navegador:  https://xxxxxxxx-5000.proxy.runpod.net
Configuração:       perfil automatico 'gpu48'
```

Abra o link e use igual no PC: arraste os vídeos e clique em **Iniciar Dublagem**.

## 3. Nas próximas vezes

Ligue o pod (**Start**) e rode só:

```bash
bash /workspace/LuminaDub/runpod/iniciar.sh
```

Liga em menos de 1 minuto. Para atualizar o código antes: `cd /workspace/LuminaDub && git pull`
(se o `requirements.txt` mudou, o script reinstala sozinho só o que precisa).

## Duas placas (ou mais) no mesmo pod

Escolha **GPU count = 2** no RunPod. O `iniciar.sh` percebe e liga **um servidor por placa**:

```
Placa 1:  https://xxxx-5000.proxy.runpod.net
Placa 2:  https://xxxx-5001.proxy.runpod.net
```

- Abra os dois links em duas abas e mande **uma parte dos cursos em cada** (ex.: NK101 + NK202
  numa, NK303 + NK404 na outra). Cada um tem a sua fila.
- Os modelos são baixados uma vez só; os dublados das duas vão pra **mesma pasta**
  `data/dublados/<Curso>/...`.
- Custa o dobro por hora mas termina na metade do tempo: **o custo total é o mesmo**.
- Pra usar só uma das placas: `DUB_GPUS=1 bash runpod/iniciar.sh`.

## Comandos

| Comando | O que faz |
|---|---|
| `bash runpod/iniciar.sh` | instala (se precisar) e liga |
| `bash runpod/iniciar.sh status` | placas, links, perfil e VRAM em uso |
| `bash runpod/iniciar.sh logs` | acompanha a dublagem ao vivo (Ctrl+C sai) |
| `bash runpod/iniciar.sh parar` | desliga o servidor |
| `bash runpod/iniciar.sh reinstalar` | apaga o ambiente Python e instala do zero |
| `bash runpod/merge.sh` | lista projetos; `merge.sh ID` refaz só o vídeo final |

## Custos e dicas

- Vídeo de 15 min ≈ **US$ 0,08** na L40S ou na A6000 → **US$ 15 ≈ 150–180 vídeos**.
- Mande vários vídeos de uma vez: eles entram em fila e os modelos ficam carregados entre um e outro.
- **Dê Stop no pod quando terminar.** Parado, só o volume cobra (centavos por GB/mês).
- Baixe os vídeos prontos pela aba **Prontos** → **Baixar todos** (.zip).
- Se o servidor cair, ele volta sozinho em 5 s; a fila continua de onde parou.

## Problemas

- **Link não abre:** confira se a porta `5000` (e `5001` com 2 placas) está em *Expose HTTP Ports* e rode `iniciar.sh status`.
- **Erro na instalação:** rode `bash runpod/iniciar.sh reinstalar`.
- **Ver o erro do servidor:** `tail -50 /workspace/luminadub-gpu0.log` (placa 2: `luminadub-gpu1.log`).
- **`faster-qwen3-tts indisponível`** no início: a voz funciona, só ~2x mais lenta — me mande o log.
