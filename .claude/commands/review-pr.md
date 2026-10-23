---
description: Verifică modificările curente conform PR_GUIDE.md
---
Citește `PR_GUIDE.md`. Argumentul este rolul meu: "autor" sau "reviewer" ($ARGUMENTS); dacă lipsește, întreabă.

Rulează doar comenzi git de citire (`git status`, `git log --oneline main..HEAD`, `git diff --stat main...HEAD`, `git ls-files`) și citește fișierele modificate. Parcurge checklist-ul potrivit din ghid (secțiunea 1 pentru autor, secțiunea 2 pentru reviewer) și dă-mi un raport cu OK / ATENȚIE / PROBLEMĂ pentru fiecare punct, plus lista a ceea ce trebuie să verific eu manual. Nu face commit, push sau merge și nu modifica cod.
