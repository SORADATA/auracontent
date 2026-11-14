# System
Tu produis uniquement du JSON valide. La cle 'scenes' contient EXACTEMENT {scene_count} scenes, ni plus ni moins -- c'est une contrainte absolue et non negociable. Ne montre jamais ton raisonnement, reponds directement avec le JSON final.

# User
SUJET : {topic}
ACCROCHE OBLIGATOIRE : {chosen_hook}
{grounding_block}

REGLES STRICTES DE NARRATION ET VISUEL :
0. La scène 1 COMMENCE MOT POUR MOT par l'ACCROCHE OBLIGATOIRE, puis enchaîne immédiatement sur un fait concret.
1. Interdiction de faire de longs discours d'introduction ("Nous allons vous raconter...").
2. Dès la scène 2, entre DIRECTEMENT dans le vif du sujet avec des faits concrets.
3. Le milieu de la vidéo doit développer l'histoire en profondeur.
4. Les dernières scènes doivent apporter une conclusion claire ou une révélation.
5. IMAGE ('image_prompt') : TU DOIS OBLIGATOIREMENT L'ÉCRIRE EN ANGLAIS. Génère une description STRICTEMENT VISUELLE ET MATÉRIELLE de 15 mots maximum (objets concrets, architecture, lumière). Interdiction absolue d'utiliser des termes de synthèse (CGI, 3D) ou des concepts abstraits.
6. LECTURE AUDIO : N'utilise JAMAIS de chiffres romains. Écris-les obligatoirement EN TOUTES LETTRES (ex: "vingtième siècle").
7. 'stock_search' : OBLIGATOIREMENT EN ANGLAIS. Ne demande JAMAIS de lieux géographiques précis. Décris en 2 à 4 mots un OBJET ou un GESTE concret lié à la phrase (ex : "old iron key", "hands opening manuscript", "gold coins candlelight"), jamais une ambiance vague ("mystery", "history", "dark atmosphere").
8. RYTHME SHORTS : le 'text' de chaque scène fait 2 phrases courtes, environ 7 secondes. Un fait concret par scène. Jamais de remplissage.
8-0. MATIÈRE : CHAQUE phrase s'appuie sur un des FAITS UTILISABLES listés plus haut. Une phrase qui n'apprend rien (« L'histoire commence ici », « Le mystère reste entier ») est interdite. Si les faits manquent, fais moins de phrases : n'invente jamais un fait pour remplir.
8-1. RÉPONSE : le script répond clairement à la question que pose le SUJET (la RÉPONSE donnée plus haut), dans l'une des dernières scènes.
8-2. IMAGES : n'écris jamais une phrase pour justifier une image (« le pont en témoigne aussi », « on le voit encore ici »). Les images sont choisies d'après ton texte, jamais l'inverse.
8a. STYLE PARLÉ, OBLIGATOIRE :
   - {max_sentence_words} mots AU PLUS par phrase. Une idée par phrase. Coupe plutôt que d'enchaîner les virgules.
   - Vocabulaire de tous les jours, comme si tu racontais l'histoire à un ami : « les moines partent », pas « les religieux quittent les lieux ».
   - AUCUN adjectif décoratif. Un adjectif n'est permis que s'il apporte une information (une couleur, une matière, un chiffre). Mots interdits : {banned_words}.
8b. IMAGES RÉELLES : la vidéo est illustrée par les vraies photos du lieu. Au moins {min_specific} scènes sur {scene_count} sont de type 'specific' : elles montrent le lieu réel ou une de ses parties (façade, cloître, salle, statue, objet) et 'location_name' nomme ce lieu exact. La scène 1 est toujours 'specific' : c'est l'image d'ouverture. Les autres scènes peuvent être 'generic' (un objet ou un geste, décrit dans 'stock_search').
9. RE-HOOK OBLIGATOIRE : La scène {rehook_scene} doit contenir une phrase de rupture qui relance la curiosité EN apportant un fait de la liste (ex : « Mais le sculpteur a gardé un autre modèle. »).
10. PHRASE FINALE : Génère un champ 'closing_cta' indépendant dans le JSON, une phrase courte (6-10 mots) qui pose au spectateur une question précise sur CETTE histoire, à laquelle il peut répondre en commentaire (ex : « Et vous, l'auriez-vous rendu ? »). Jamais « abonnez-vous », jamais « likez », jamais d'annonce de la prochaine vidéo.
11. Les clés 'mood' doivent être choisies parmi : "ominous", "intriguing", "tense", "awe", "scientific", "investigation".
12. Les clés 'role' doivent être choisies parmi : "hook", "tension", "context", "value", "escalation", "reveal", "cta".

FORMAT DE SORTIE ATTENDU :
Retourne un JSON avec les clés : title, on_screen_title, era, visual_identity, audio_profile, closing_cta, scenes.
'era' : l'époque du fait raconté, EN ANGLAIS, courte et précise (ex : « late 18th century », « 1790s », « 12th century »). Elle sert à dater les images.
'on_screen_title' : le bandeau affiché en gros pendant TOUTE la vidéo, et lu sur la miniature. Format obligatoire « CONSTAT | CHUTE » : deux segments séparés par une barre verticale, 2 à 4 mots et 18 caractères au plus par segment. Le premier pose le fait, le second le renverse ou le chiffre. Le bandeau contient un verbe (« VENDUE », « DISPARAÎT ») ou un chiffre, et au moins un chiffre, une date ou un nom propre PRÉSENT DANS LA SOURCE ; une date ou un chiffre n'est permis que s'il figure mot pour mot dans la source ; jamais une simple liste de noms de lieux. Compréhensible sans le son et sans avoir vu la vidéo. Pas de question, pas de ponctuation finale (ex : « MONT-SAINT-MICHEL | 70 ANS DE PRISON », « CARCASSONNE | PROMISE AUX DÉMOLISSEURS », « LA JOCONDE VOLÉE | SOUS UNE BLOUSE »).
Chaque scene dans le tableau 'scenes' doit contenir : id, text, voice_direction, pause_after_ms, stock_search, image_prompt, location_name, location_country, voice_type, mood, role, scene_type, event_context.