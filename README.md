# Predviđaj – GitHub Pages

Primenjuje **već istrenirani** Ridge/linearni model na nove podatke vage, direktno u browseru. Nema treniranja, servera, CDN-a ni biblioteka.

## Objavljivanje
1. Novi **public** repozitorijum (npr. `predvidjaj`).
2. U root postavi `index.html`, `style.css` i `app.js`.
3. Settings → Pages → Deploy from a branch → `main` → `/(root)`.

## Korišćenje
1. Na sajtu za treniranje (pcelarska-vaga) učitaj log i klikni **Preuzmi Ridge model (JSON)**.
2. Ovde učitaj `vaga_ridge_model.json` (korak 1) i nove podatke `vaga.txt` (korak 2). Redosled nije bitan.
3. Prikazuju se izmerene vrednosti kakve jesu, uporedo sa predviđanjem i kompenzacijom.

## Kako radi predviđanje
- Temperatura novog loga se računa u odnosu na referentnu temperaturu iz treninga (`ref_temp`), pa se prave isti 9 ulaza kao u treningu.
- Model daje **temperaturni drift u gramima**. Pošto apsolutna težina novog loga nije ista, drift se usidrava na prvih 4 merenja (1 h): predviđanje = drift + prosek (izmereno − drift) na tih 1 h.
- Kompenzacija = izmereno − predviđeno. Nije garantovano stvarna promena mase.
- Prvih ~6 h loga nema pune istorije za Ridge (isto kao na početku treninga).
- Ako je temperatura van opsega treninga, sajt upozorava da je predviđanje ekstrapolacija.
- Log mora biti 15-minutni, bez rupa i duplikata, najmanje 30 merenja.

## Privatnost
Fajlovi se čitaju samo lokalno. `vaga.txt` ne postavljaj u javni repozitorijum.
