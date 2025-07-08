# TesseractOCR Test

Two paramter configurable : PSM Mode and Image Enchancement Level

## PSM mode 



## Test methodologie

1. Test the PSM mode influence. Fix enchancement level to MEDIUM
2. Test enchancement level influence. Fix PSM mode to SINGLE_BLOCK
3. Analyse for first result.
4. Test combinaisons.

## RAW Test Results (Test 1 and 2)
```shell
----------------------------------------
Using PSM mode: AUTO_WITH_OSD
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
d Microwave le)

Radio Frequency an
----------------------------------------
Using PSM mode: AUTO_WITHOUT_OSD
2025-07-08 14:06:21,228 - ERROR - Text extraction failed: [Errno 2] No such file or directory: 'C:\\Users\\JULIEN~1\\AppData\\Local\\Temp\\tess_zlux8do4.txt'

----------------------------------------
Using PSM mode: AUTO
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
d Microwave le)

Radio Frequency an
----------------------------------------
Using PSM mode: SINGLE_COLUMN
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
d Microwave le)

Radio Frequency an
----------------------------------------
Using PSM mode: SINGLE_BLOCK_VERT_TEXT
o
c
c
£&
fe)
=
s
oo
2 -
=°
_ ©
=
moO
2 §
z=
32
2s
£ Ss
As°)
c
a3
§ &
fo}
S
----------------------------------------
Using PSM mode: SINGLE_BLOCK
saotharlann Radaimhiniciochta agus Mhicreathoinne
Radio Frequency and Microwave Lab
----------------------------------------
Using PSM mode: SINGLE_TEXT_LINE

----------------------------------------
Using PSM mode: SINGLE_WORD
=
----------------------------------------
Using PSM mode: CIRCLE_WORD
ciao
----------------------------------------
Using PSM mode: SINGLE_CHARACTER
ciao
----------------------------------------
Using PSM mode: SPARSE_TEXT
Mhicreathoinne

Saotharlann Rad

gimhiniciochta agus

le)

Radio Frequency an

d Microwave
----------------------------------------
Using PSM mode: SPARSE_TEXT_OSD
Mhicreathoinne

Saotharlann Rad

gimhiniciochta agus

le)

Radio Frequency an

d Microwave
----------------------------------------
Using PSM mode: RAW_LINE
=
----------------------------------------
Using Enhancement Level: LIGHT
Saotharlann Radaimhiniciochta agus Mhicreathoinne
Radio Frequency and Microwave Lab
----------------------------------------
Using Enhancement Level: MEDIUM
saotharlann Radaimhiniciochta agus Mhicreathoinne
Radio Frequency and Microwave Lab
----------------------------------------
Using Enhancement Level: STRONG
* ieee a,
-_ aan rae et, x ay Ok “Gees ahah “ o Datanee “th erie
SAP ABR ata we Petts 4 mas “elves VS Nets eh 8 ato fanta(l
rere Agieg oy OER Duy UR Sy SIS BAYES yuna seis ORDO tian ie wey
a cS DAs Yee oes es lhe yee whe cy gical Sa nun) sy fait part ae tigen ae
OMENS. poe NES MT prey goth, a4 UES Pape ae th Sheth t Ba craene rer Ri Ae
ae Mente ee a ro Wee ke RS Ty whe £ Aiwtordg eat ae an Band ae ee a
“ithe 25) Sate ae ane Se gens Sate ee gine 3 TERRA Rae Hil ee
Se gent ri Le: Cle 3 a Phe
ny i ids ey Sy Saree y “Anaya eh eh OES Me ae
gay PY ere gett) Mahle tery 22 uh MPa et Penis oe) ae,
ee Tho hat EM ee hick whee Re Nees EEDA eae cio ORS) 182 ai
eee | Crna as a Ea CORES | aan eae tee Ceaighe ite sh RAS Be A te
ae NE ace tajagu a i
veetic aoe oasis a.
ease imniniciochta WENS ai
et ae ee daimhi NO a ear ee
a Sa rt 2 Ee planets Hut) HEINER We
ran . Freq Rakin Vines SG hee 3 erent
2 SNP Ae EN SR AER FAVRE are Sint
op Radic too
oy Av eh sap 2 cele: DRAB Hast ataals nt
a er ED, SA Neal ee ele Seer asa (eet Ate
va There rue miei TEAS 5 Sot ERS wages eR ALI Were
HY Se cert Mer Ba: xP in dee Peete Sy SS Ret Ra a
at = TASES Pee Shines MACRO poor Pease Sienna Bee SAS SEE :
SY PTR Oe Rs PASS PES ES Hees ve BEERS SN Crs Sey ASROU Sigates Say
en a are cet Pretec tiainict 23 RSS ext bee acral aL ‘I Sa een si ties
yy ae an aoerk tara eee +8 Hosingerg hase. Pe SE ONS es ote Sis PEE
eran ae Fans a Eames at ERAT Si Eons: ALES oe ete sah mae
Syed. “eH oe SES SUS GEN aa ores. ae Roa a
Gus ea" ge ae PES AR Se ah Saag LANiRAR
Ail yok Sia flay iT Navn oe EN AC ARLENE
Pee we ae Bret at eRe Lees meee ti ue! iets Rye sey Esehieat Rs aN
iy, We SS Rees Ue es oes Hs abe
----------------------------------------
```

## First Analytics
The PSM mode SINGLE BLOCK - on the sign image - is the best.
With a STRONG enchancement processing image we have bad results, a new word and lines was generated.
The LIGHT enchancement processing give the best result. He detectect the first caps lettre compared to MEDIUM enchancement processing.

## Combinaison test

1. All PSM Mode with LIGHT enchancement processing.

```shell
RESULT :
Using PSM mode: AUTO_OSD
2025-07-08 14:20:49,026 - ERROR - Text extraction failed: (1, 'Warning, detects only orientation with -l eng Error, OSD requires a model for the legacy engine')

----------------------------------------
Using PSM mode: AUTO_WITH_OSD
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
nd Microwave Lab

Radio Frequency a
----------------------------------------
Using PSM mode: AUTO_WITHOUT_OSD
2025-07-08 14:20:49,851 - ERROR - Text extraction failed: [Errno 2] No such file or directory: 'C:\\Users\\JULIEN~1\\AppData\\Local\\Temp\\tess_ttff_b5i.txt'

----------------------------------------
Using PSM mode: AUTO
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
nd Microwave Lab

Radio Frequency a
----------------------------------------
Using PSM mode: SINGLE_COLUMN
Mhicreathoinne

gimhiniciochta agus

Saotharlann Rad
nd Microwave Lab

Radio Frequency a
----------------------------------------
Using PSM mode: SINGLE_BLOCK_VERT_TEXT
o
c
(S
=
(eo)
cS
oS
S| Tes)
2
=°
aS
=
= 6
2 5
=
82
2g
5
ES
al fey
et
[S
& 3
eS
fo}
C=)
----------------------------------------
Using PSM mode: SINGLE_BLOCK
Saotharlann Radaimhiniciochta agus Mhicreathoinne
Radio Frequency and Microwave Lab
----------------------------------------
Using PSM mode: SINGLE_TEXT_LINE

----------------------------------------
Using PSM mode: SINGLE_WORD
=
----------------------------------------
Using PSM mode: CIRCLE_WORD
ei eae
----------------------------------------
Using PSM mode: SINGLE_CHARACTER
ei eae
----------------------------------------
Using PSM mode: SPARSE_TEXT
Mhicreathoinne

Saotharlann Rad

gimhiniciochta agus

Lab

Radio Frequency 4

nd Microwave
----------------------------------------
Using PSM mode: SPARSE_TEXT_OSD
Mhicreathoinne

Saotharlann Rad

gimhiniciochta agus

Lab

Radio Frequency 4

nd Microwave
----------------------------------------
Using PSM mode: RAW_LINE
=
----------------------------------------
```

The PSM Mode SIGLE_BLOCK is the best with LIGHT enchancement level.

I have test LIGHT, MEDIUM and STRONG enchancement level with AUTO_WITH_OSD PSM mode but the enchancement mode degrade the result.

## Conclusion

Use SINGLE_BLOCK PSM Mode with a LIGHT enchancement level processing.