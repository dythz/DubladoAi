# LuminaDub AI

Dublagem automática de vídeos **do inglês para o português do Brasil**, rodando 100% no seu computador, de graça.

Você arrasta as aulas no site, e ele:

1. separa a voz da música de fundo (Demucs)
2. transcreve o inglês (Whisper)
3. traduz para português do Brasil (Opus-MT ou Ollama)
4. gera a fala em português **com a voz de quem fala no próprio vídeo** (Qwen3-TTS, acelerado com faster-qwen3-tts)
5. sincroniza com o vídeo e devolve o vídeo dublado, com legendas PT e EN

Leva mais ou menos o mesmo tempo que a duração do vídeo (numa RTX 3060).

## Do que você precisa

- **Windows** com uma **placa NVIDIA** (testado numa RTX 3060 12 GB; 8 GB deve funcionar, mais lento)
- **Python 3.11** — `winget install Python.Python.3.11`
- **FFmpeg** — `winget install Gyan.FFmpeg`
- Uns **15 GB livres** em disco e internet na instalação

## Instalação (uma vez só)

1. Baixe o projeto: botão verde **Code → Download ZIP** e extraia, ou `git clone` deste repositório.
2. Dê dois cliques em **`instalar.bat`**. Ele cria o ambiente Python, instala as bibliotecas e baixa o modelo de voz (~4 GB). Leva de 10 a 30 minutos.

## Uso

1. Dê dois cliques em **`start.bat`**.
   - Ele pergunta **"Limpar tudo?"** — aperte **N** (ou espere 8 segundos). **S** apaga todos os projetos e vídeos dublados.
2. Abra **http://localhost:5000** no navegador.
3. Arraste os vídeos, deixe **Qwen3** e **Voz do vídeo** marcados e clique em **Iniciar Dublagem**.
4. Os vídeos prontos ficam na aba **Prontos** (dá para baixar um por um ou **Baixar todos** em .zip).

Pode colocar vários vídeos de uma vez: eles são dublados um atrás do outro. Se o computador desligar no meio, ao abrir o `start.bat` de novo ele continua de onde parou.

## Dicas

- **Placa com menos de 12 GB:** funciona, mas se ficar lento, feche jogos e outros programas que usam a placa enquanto dubla.
- **Atalhos no player:** espaço (play/pause), ← → (10 s), S (velocidade), C (legendas), F (tela cheia).
- **Edge** (no Motor de Voz) usa vozes prontas da Microsoft: mais rápido, mas precisa de internet e não clona a voz.
- `baixar_videos.bat` copia todos os vídeos dublados para a sua pasta Downloads.
- `reset.bat` apaga todos os projetos (pede confirmação).

## Problemas comuns

- **"venv nao encontrado"** ao abrir o `start.bat`: rode o `instalar.bat` primeiro.
- **Erro na instalação por internet:** rode o `instalar.bat` de novo, ele continua de onde parou.
- **A janela preta fechou sozinha / o site não abre:** abra o `start.bat` de novo e veja a mensagem de erro na janela.
