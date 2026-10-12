# Regole di progetto (lette a ogni turno)

## 1. Economia dei token (priorità assoluta)

La sessione principale gira sul modello più costoso e rilegge tutto il contesto a ogni turno. Il dettaglio e il perché sono in `docs/TOKEN_ECONOMY.md`.

1. **Ogni chiamata Agent porta `model`**: `opus` per il lavoro, `sonnet` se Opus risponde con errori server, `haiku` per ricerche banali. Mai un agente senza `model`.
2. **Un risveglio per lavoro, non per evento.** Niente `Monitor`, niente cron di polling. Un driver bash in background esegue tutta l'attesa e termina (`training/fleet/wait_wave.sh`, che a fine onda fa anche il pull); oppure un agente aspetta con cicli `until` in primo piano (≤10 min ciascuno) e riporta una sola volta a fine lavoro.
3. **Mai leggere artifact, transcript o file grandi nella sessione principale** (log di training, eventi TensorBoard, checkpoint): se serve una lettura, la fa un agente e riporta i numeri.
4. **La contabilità è delegata**: NIGHT_LOG, WAVE_PLAN, tabelle delle onde (`training/logs/waves/`), memoria: un solo agente per passaggio, la sessione principale rilancia il rapporto.
5. **Risposte brevi.** Nessun messaggio di conferma per eventi di routine; dopo un output grande suggerire `/compact`.
6. **Non controllare un training più spesso di ogni 500 iterazioni** e restare inattivi nel frattempo.
7. **I driver non devono richiedere un modello**: lancio, attesa, ripartenze, pull dei file: tutto in bash (`training/fleet/fleet.sh`, `training/fleet/supervise_train.sh`, `training/fleet/wait_wave.sh`).

## 2. NIGHT_LOG.md: scritto per essere letto dall'utente

Il diario serve all'utente per capire in un minuto **dove siamo e perché**, non per ricostruire cosa hanno fatto gli agenti. Regole:

1. **Orario dal sistema, mai a mano**: ogni riga inizia con `$(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M CEST')` preso al momento della scrittura.
2. **Una riga per evento che cambia qualcosa**: lancio di un'onda (cosa e perché, in una frase), verdetto di un'onda (la run migliore e i numeri decisivi), nuovo deliverable (nome, iterazione, percorso), fallimento o blocco che richiede una decisione. Nient'altro.
3. **Non scrivere**: le righe per singolo checkpoint o per singola run (stanno nelle tabelle in `training/logs/waves/<onda>.md` e nelle note `training/logs/agents/*.md`); i dettagli dei driver e dei bug di infrastruttura risolti; le note operative sugli agenti; le ripetizioni di numeri già scritti.
4. **Lunghezza**: al massimo 3 righe (≈ 400 caratteri l'una) per onda: lancio, verdetto, decisione. Se un'onda ha 4 run, il verdetto è una riga con la vincitrice e il conteggio delle altre ("1 successo su 4, 2 arrivano in fondo sopra i 2 cm"), non quattro righe.
5. **Frequenza**: non più di una riga ogni 30 minuti salvo un blocco. Fra due eventi il diario tace.
6. **Numeri decisivi soltanto**, sempre nello stesso ordine e con la base accanto: valutazione deterministica dal frame 0 (`training/scripts/evaluate.py --start 0`) come "frame raggiunti/828, palmo medio cm, RMS mano rad", poi dal training la frazione di terminazioni anticipate e la fase finale. Gli altri numeri vanno nelle tabelle.
7. **Percorsi**: ogni deliverable o candidato nominato porta il suo percorso `training/logs/motion_imitation/...`.
8. **Lingua**: italiano, frasi brevi, verbi al presente; niente prosa di rassicurazione.

Esempio di riga giusta:
`- **2026-10-12 14:55 CEST** stage 1 desktop, verdetto: model_2999 dal frame 0 termina a 27/828 (dal 700 arriva in fondo a 2.9 cm); training terminazioni 0.31 (da 0.99), fase finale 0.90, palmo 3.5 cm. Non converge: serve più campione. Run in training/logs/motion_imitation/2026-10-12_14-01-03_stage1.`

## 3. Vincoli operativi

- Mai uccidere processi dell'utente (teleoperazione, viewer, tutto sotto `/home/simone/.venv` o `teleoperation/`); mai aprire le telecamere né toccare lo stimatore della teleoperazione; mai `sudo`; solo docker rootless.
- Sui server: mai toccare `~/SimToolReal*`, `~/simtoolreal_newton`, `~/simtoolreal_animrl_ee`, `~/docker_*` né container il cui nome non inizi con `bim_`; mai toccare job GPU di altri utenti (`fleet.sh` rifiuta una GPU con > 1 GB occupato).
- GPU del desktop (condivisa con lo schermo): al massimo 1024 env.
- Interprete: `env -u PYTHONPATH deps/IsaacLab/.venv/bin/python`.
- Vocabolario in `CONTEXT.md`, decisioni in `docs/adr/`, piano delle onde in `WAVE_PLAN.md`.
