# Pčelarska vaga – GitHub Pages

Interaktivna analiza `vaga.txt` **direktno u browseru**. Nije potreban Python, server, CDN, npm ni internet nakon učitavanja sajta.

## Objavljivanje na GitHub Pages

1. Napravi novi **public** GitHub repository (npr. `pcelarska-vaga`).
2. U **root** repozitorijuma postavi `index.html`, `style.css` i `app.js`.
3. Otvori **Settings → Pages**, kao source izaberi **Deploy from a branch**, branch **main**, folder **/(root)**, pa **Save**.
4. Sačekaj da GitHub objavi sajt. Adresa će obično biti `https://KORISNICKO_IME.github.io/pcelarska-vaga/`.
5. Na sajtu klikni **Izaberi fajl** i izaberi svoj lokalni `vaga.txt`.

Možeš ga pokrenuti i lokalno prostim otvaranjem `index.html` u browseru, bez internet veze. Nema mrežnih poziva; log ostaje u browseru korisnika.

## Format loga

Jedan red:

```
15-10-2026 21:00:00|117.189g|+/-0.611g|117.153g|+/-1.111g|15.625C|4.143V
```

`datum|tezina1|greska1|tezina2|greska2|temperatura|baterija`

- Datum `DD-MM-YYYY HH:MM:SS`.
- Log može biti **rastući, opadajući ili mešovit**: datumi se uvek sortiraju hronološki pre treniranja, a referenca je najstarije merenje.
- Interval **15 minuta**, bez duplih datuma i preskočenih termina; kod prijavljuje precizno mesto prekida.
- Nepotpuni / neispravni redovi preskaču se, ako preostalih ispravnih merenja ima najmanje 48 (12 h; za ozbiljnije rezultate koristi što duže logove).

## Računanje

- Linearna regresija: `delta težina1 ~ delta trenutna temperatura`, sa interceptom.
- Ridge 6h: 9 ulaza – T sada, T pre 1h/3h/6h, srednja T poslednjih 3h/6h, EMA 30/120/360 min. 25 merenja 15-min temperature za lagove; EMA pamti i starije.
- Standardizacija odgovara sklearn `StandardScaler`; Ridge L2 kazna odgovara sklearn `Ridge` (bez regularizacije intercepta).
- Hronološki test: prvih 70% trening, zadnjih 30% test. `alpha` bira 4-vremenski TimeSeriesSplit na treningu (0.001, 0.01, 0.1, 1, 10, 100).
- Grafik prikazuje **nove modele trenirane na CELOM logu**, a RMSE prikazuje performanse modela treniranih samo na prvih 70%.
- Reziduali (`izmerena − predviđena`) nisu nužno fizička promena mase. Kalibracija važi samo ako je masa bila konstantna tokom treninga.

## Funkcije

- 3 interaktivna grafikona sa checkboxovima i sinhronizovanim horizontalnim zumom:
  - Težina1, **Težina2**, temperatura, linearno predviđanje, Ridge predviđanje (g i °C)
  - Linearna i Ridge kompenzacija (g)
  - **Greška1, Greška2 i napon baterije** (posebne ose ±g i V)
- **ML treniranje ostaje samo na Težini1 i temperaturi**, drugi parametri se samo prikazuju
- Tooltip sa datumom i odgovarajućim jedinicama (g, °C, V)
- Točkić miša za zum, izbor perioda prevlačenjem i reset zumiranja
- Export PNG za svaki grafikon, CSV sa **svih šest izvornih veličina**, predviđanjima i kompenzacijama
- Boje: **Težina1 tamnoplava**, **Težina2 svetloplava**, **temperatura crvena**, **linearno žuta**, **Ridge crna**, Greška1 zelena, Greška2 koralna, baterija ljubičasta

## Izvoz modela

Dugme **Preuzmi Ridge model (JSON)** čuva istrenirani Ridge i linearni model (koeficijenti, standardizacija, alpha, referentna temperatura) u `vaga_ridge_model.json`. Taj fajl se učitava u zasebni sajt **Predviđaj** zajedno sa novim podacima.

## Privatnost

Nijedan log se ne otprema na GitHub ni bilo koji API. Upload polje samo lokalno čita izabrani fajl preko JavaScript `File.text()`. Statički sajt i njegov izvorni kod mogu biti javni, ali **vaga.txt ne treba uploadovati u javni repozitorijum**.
