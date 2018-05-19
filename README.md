# char_wgu

*char_wgu* is a minimal character level LSTM/RNN model that learns from sequences of text from a given file and generates text sequences based off of what is has learnt from the input text.
This project draws its inspirations from Andrej Karpathy's phenomenal post, [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). 
*char_wgu* is built with Tensorflow, and support for the latest releases is a major priority. 


## Samples generated with char_wgu
After not-very-long training on [Google Colaboratory](https://colab.research.google.com) on moderately sized sequences of texts, the model generates surprising sequences.
Some randomness is introduced into the output sequences by changing the sampling temperature. See samply.py for mode details.

* I trained a model on the text of the Constitution of Nepal 2072BS. The text was relatively small (about 200k characters) and the model was trained for 20 epochs and it took about 130 seconds per epoch. 

```
﻿ योग्यता नभएमा वा नरहेमा,
(ग) प्रदेश सभाको कार्यकाल विशेष सभामा फिर्ता पठाएकारमा निर्वाचन गरी नेपालको समटक्षाये, पण्त्रजुनन्ज हुत र सम्पत्ति प्रसहितुर्पाी र समुत्यक, प्रएको जनड्डहण्त्रिाको दलित पार्रि पूरा भएको, 
(ग) नियुक्ति हुँुपारिणा ऋ२प्रति निखित, प्रत्येक वह्यहनका लागि किप÷ालि संसतगरेको पर्छ
ह्यापा. आध्यक्ष महिला हितसण्¥य रहनेछनरू तरीका ङ

९ठ कुनआले संख्या सचघ्न ता प्रकारको आैिसऋ” भन्नात्त वा नि२प्ति र चार४ हुने निर्वाचन परिव५–को त्यस्तो संवठ्ठापन संघीय संस।ले सख्यार
ठढडण्ण्ष्त ृण्ण्स्ण्ट, ज्ञद्दण्घढ।घढष्तरके                                                                                                                                                 
ेवानिवृत्त भएपछि कुनै पनि अड्डा अदालतमा बहस पैरवी,
मेलीले अलााकरताश्त्तट व्या९, लगानी, अड्डिले आय अनुखक, अत्या। निर्मावसरको पढप. गर्नु पर्नेछ ।
भग–२
न भएको पन्५ञ द्वनुक्ष वोम्भानियुमितिको अेिजिवाया संघीय संसश संकटकाल घोषणा भएको मानिनेछ {1}।
ख्याराष्ध्यिरू ग गmण्मान र अण्ुसार भेरथ्यो विराष्ज्ञ«य गुाप्रघ्नको निर्वाचन उपयोग )थाmणे संश्र–रट (च) प्रवितिक कोषबाट हुनेछ ।
म्भानियुमितिको अेिजिवाया संघीय संसश संकटकाल घोषणा भएको मानिनेछ ।
ख्याराष्ध्यिरू ग गmण्मान र अण्ुसार भेरथ्यो विराष्ज्ञ«य गुाप्रघ्नको निर्वाचन उपयोग )थाmणे संश्र–रट 
(च) प्रवितिक कोषबाट हुनेछ ।
भाछनो मात्र भएको जिल्ला मभहको निर्वाचनका द्योह्यो भाग०
संघ, प्र राष्म्भाहन नागरिक
प्रअह्वान गर्नेछ ।
२९१. संवएको अन्य काम,
कर्तव्य, अूिहलाई राष्.
विषयमा स्थापना व्यवस्र्थाि संकटव्यका नीति (था  फ्नो स९भ्न नेपालको टधार २०ब्तर्
ित्तढ द्यएला विऐ मतस्ताको व्यवस्था ुपार्४े
छण्श्न पृिन्त सर्मु
िुई तिहाइ बहुमतले अनुमोदन गरेमा त्यस्तो घोषणा वा
आङ्गीचकोअयोगताबाट ब्णिारको संख्याः गठन ग)ी सम्बन्÷ित सेवा सम्बन्२ह्न गरिनेछ ।
(४) राष्ींक्न स४घो प्रद्वान फुसला भएका व्यवस्थापिकाड संकटकालीन अवस्थाको पारिछ सेवाको प–रु२को अेिवेक्त अढा४ हुने अघि ज–न पूल हुनेछ ।
(९) निजो माफर्ण परेको हुनेछ ।
२९५. संविट विश्रचिकले नेपालको प्रत्येक प्र;क्न वा
अँिच वकि सेवाको त्यस्तो संस्था संकटकालीन अवस्थाको
ल्नुचित हुनेछ ।
(३) यस ञ्पखीस, mनो किूङ्गि द्द३ण ३ज्ञाट जुस ऋश्र मर नेपालको नेपालको नागरिक सब्स्तामा एक राष्जित अनुसनर््ाय, आयोग

(३) दुई वा दुईभन्दा बढी प्रदेशले अनुसूची–६ मा उल्लिखित जानकारी नेपालको नागरिकता प्राड्डछ भन्टेको व्यवस्थापिकाञ । र
स्थापिँ्नो रकम, जसमन्त्री
– व्यवपालिका पद्यकारन वा
बज। बनाट्टबाट हुने छइठ ।
वि)ास वा जन्न हुनेछ ।
(य) यो संवि५इ अन्य द्योषलाथा व्यवस्थापिकाईमा पाँचन गरेको जनसभामा नेपालका नागरिक धर्म, घ्राज्या्रझ्न आवश्यक पर्ने
```

*I really like how the model decides to declare a state of emergency at {1}.* The model seems to have learnt to use lists in no particular order. The grammar makes little sense, but some words are used in hilarious ways.

* Another one was trained on the entire text from the Harry Potter series, including Tales of Beedle the Bard. 

```
Tt launched a new attack on Harry. “We need time magic oé it ever dropped a pot though a dencence o— driptaned upan eethaving to beginS to let me deWorve one ochoo has, according evaled proto working and herotule’s rubbittach, then are stood with her and con have suverated which is her Lupersain, oS @ary ” and où having heart, and o-locket.  suverouted to verd admit me between oeths Eingucant none! … is the Jardity wextrees, because though, peréacted into the lubbed in in: His years, and its away took in cugh. What dogict is trair’s pince, to it demblited, this nearer constressers who had heard where an exnementatian door, the hitch stepped abit or notice. 1ß“Who becalled a mack to tell it has also © babbitty Qubrerol once’s ready.
[Okan!” asked one, rasped more, very and he crettered with either to topple their wicdows would be intempaned? Sir to be times more years,” said Year. “Interder ozen by top babbitty, so it will realys that witches, and claimed their asking into their rushour gripping Beeth oqally 'elb not.
```
This one is an intersting soup. We won't replace JK Rowling anytime soon, that's for sure. 
The best part for me is how the model learnt to open and close quotes properly, and also how it properly structures a speech. The names it has invented are pretty much what you'd expect from a Harry Potter book too. 


* This one is from a model trained on all of Arthur C Doyle's works. The grammar is pretty decent in this one, and the model seems to have created some adjectives of its own.

```
"You make it."
  "I am so acted clearly. What he bit indeed we should not really with break
n A°ricloon. Hulp, for me the medy certainly he give a affairs, sir. It would raid a man as mard of accused for
  the sout ssinstor and his bedue (worn of *uner humbrelow and alonoly with. I was alone witwin
  patient. There is a short relien. It can relet it
  of pulsir; and there has no-eflar over, heard there
 sonent undid you shall you, the dig of this profer each the centre of them," said Holmes.
  "What havea; we more beare accounted," said Holmes as he
  serely to make a spart, retulling. I can stirling that passing
  police is could note arrive itself, but I in then, to two remarkable
  way. There was sucking a xo; looking up of the Bellammeshe with a
  menyes, that he am alsoing in the breating of comes of him."
 "It was aborting that the diSlenntoan of matter in a reach to vice that. He was
  to had hands an obare only pretured to monet each. You should seem in their true, I we seathed the öebroCing face would sat that
  so. The man maid for his redner and little to the mood. He will," said Holmes.
I His business. "It was weàtly glace to black the soar7.
  "ished the mircal tave any indighting of an ipper with easiligins out of rightman to me. Syve cleor a sake of there is
  before do left for the ½egson of lich--and the truthor, &' xing the leaveîon Ambers
  had a nigon by the retuln0ment. . What can he thank that to mile the liar
  than came the man.
  "Sir You was one, and that was our crely which rad disentry arrese than the
  oldrang one wants are to emper this man were sance. That
  sire thisvy immediate waves en5m, whisper[some to go them."
  "Theret arouse, Mr. Holmes. It was the now, Watson and seems one
  with our edderly his case and trying.
  "But with the slaw weals immastened that how somethings.
  I was missing on his resuld so."
  "Whenots all, traiting what he had a day brown poess. But indackun
  yet me, he far some surfig° and his fistacthor at left.
  "Perhaps confidence a path and with his goodwiman gards flocked upon
  us misten with Dragons creasully at twenty, and askeded
  the cleartenges, singonly wetarned. "It was the terrible
  Jragerz'st confidences as
  great dortened. There are knows pothes° before I don't not be a min*t you wire it
  was a more note! What dangings on the door.
  "It wretter dreadful eyes my upon to your resull in that eKpler. He were
  not be aware an an7inger and the poor door openuces Condey of the both. Sir
  man impossible it would not amart adPressan.
  Mr. When. I will nevere indessanal was to the dark loom, he
  wetered the had had tried over eleven with his wife of
  her hand o4tile whom wident cleft once will time so turn
  of kircobin told my tall, and they he had been asGood for
  a warn and strange great compenitwish which ome little blittleress ?o
  panting out this firir of an tetting down--with ask
  satisforent, the estander engless was raccural note."
  "Yes,- his endward for a constancy for ghesses, and in they
  was heavy seen to ancuon. The one of the telegrope
  caseing ancing susge of Kan-card then some dispositiver you, Mr. he
  ray was criem with the claw clie which others half and an
  this sense-hard piech true look one instets as he
  were suddenly creet in a vicaurars, and Zo could all the
  flow more man sures ; to puft interry of yourselves."
  "Let you thought we thinks at ourselves has justly a firmlance."
  Let a boller plenged out upon which fell over. But his troom's mose time sinosal
  bright brick and more."
  "I moved at the listation.
  'I saw the cloching nation for the reupuninstlon5s for an particular-bock
  that I am not really the man the hand, so huloped minnisy
  rood as Beritson. As in the ancient he master itself a
  wortys and this mernicas' may the dread my immersion Kest it.
  "Well, sir no epime
```

* More examples coming pretty soon. Lemme get through my exams first.


## Here's a few things that might help you.
* Keep the learning rate to about 10e-3 during training. That is the default learning rate used in the training script. Anything higher than 10e-2 is too adaptive to newer batches and the model tends to regurgerate whateven text was in the training examples it was trained on towards the end. 
* Train on a decent machine with a supported CUDA GPU. Training on a machine with a 12GB K80, 12GB RAM and an 8-core Xeon Phi was about 30x faster for me compared to my Core2duo machine with a 4GB RAM and no GPU. I trained my models on Google Colaboratory, which gives you free access to an IPython notebook running off of a virtual machine with the specifications mentioned above, for 12 hours at a time. The instances also come with popular libraries frequently used in machine learning pre-installed.
* A model with 3 layers of LSTMs with 128 hidden units each is more than enough for most use cases. Anything more than that is very hard to train in my experience.
* Use decently sized text files as input. Anything around 1-2MBs in size is pretty good for a taste of how effective these models are, but of course, the more the merrier.
* The training time increases with increases in
..* number of layers of LSTM cells
..* steps in time to compute gradients
..* number of hidden units in eacdh LSTM layer

## TODO
* Unit tests. UNIT TESTS. **UNIT TESTS**.
* Comment the code properly. Some blocks are too convoluted to comprehend.
* Randomize the training examples. This should also help with the overfitting issue with large learning rates.
* Add support for other RNN cells like GRU cells.
* Add live loss plot within the IPython environment for Colab.


## Known Issues
* The progress bar is rendered twice during training, the first one stopping after the other one begins.
* The progress bar overflows during sampling.
* The hidden state of the LSTM cells *has* to be preserved to minimize redundant calculations. Sampling is embarrasingly slow at the moment due to the need for calculating 49 redundant steps each time a new character is sampled. That is almost shameful.