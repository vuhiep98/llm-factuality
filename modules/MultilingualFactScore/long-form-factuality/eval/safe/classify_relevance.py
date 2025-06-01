# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Identifies whether an atomic fact is relevant for ansing the prompt."""

from typing import Any

# pylint: disable=g-bad-import-order
from common import modeling
from common import utils
from eval.safe import config as safe_config
# pylint: enable=g-bad-import-order

SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'

_PROMPT_PLACEHOLDER = '[PROMPT]'
_RESPONSE_PLACEHOLDER = '[RESPONSE]'
_STATEMENT_PLACEHOLDER = '[ATOMIC FACT]'

_RELEVANCE_FORMAT_EN = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Quoc is well-respected by many \
of his peers, such as Geoffrey Hinton, who is an adjunct professor at the \
University of Montreal and teaches courses on deep learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. The phrase "Quoc is well-respected by many of his peers, such \
as Geoffrey Hinton" from the RESPONSE shows that the relationship between Quoc \
Le and Geoffrey Hinton is that they are peers. For this reason, the subjects \
Quoc Le and Geoffrey Hinton are [{SYMBOL}].


Example 2:
QUESTION:
Who is Quoc Le?

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Geoffrey Hinton is an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

STATEMENT:
Geoffrey Hinton is at the University of Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and \
Geoffrey Hinton are [{NOT_SYMBOL}].


Your Task:
QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_REVISE_FORMAT_EN = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns is a company.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns is a company.
```


Example 2:
STATEMENT:
He teaches courses on deep learning.

RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been \
working on a variety of deep learning projects. Le is also an adjunct \
professor at the University of Montreal, where he teaches courses on deep \
learning.

REVISED STATEMENT:
The subject in the statement "He teaches course on deep learning" is "he". \
From the RESPONSE, we can see that this statement comes from the sentence "Le \
is also an adjunct professor at the University of Montreal, where he teaches \
courses on deep learning.", meaning that "he" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le teaches courses on deep learning.
```


Example 3:
STATEMENT:
The television series is called "You're the Worst."

RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her \
television debut in 2016 with a guest appearance on the Disney Channel series \
"Raven's Home." She has also appeared in the television series "You're the \
Worst" and "Gentefied."

REVISED STATEMENT:
The subject of the statement "The television series is called "You're the \
Worst."" is "the television series". This is a reference to an unknown entity, \
since it is unclear what television series is "the television series". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "the television series" is a \
vague reference that should be replaced by "the television series that Xochitl \
Gomez appeared in". Thus, the revised response is:
```
The television series that Xochitl Gomez appeared in is called "You're the \
Worst."
```


Example 4:
STATEMENT:
Dean joined Google.

RESPONSE:
Jeff Dean is a Google Senior Fellow and the head of Google AI, leading \
research and development in artificial intelligence. Dean joined Google in \
1999 and has been essential to its continued development in the field.

REVISED STATEMENT:
The subject of the statement "Dean joined Google" is "Dean". From the \
response, we can see that "Dean" is the last name of "Jeff Dean". Therefore \
"Dean" is a non-full name, making it a vague reference. It should be replaced \
by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean joined Google.
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""
#############
_RELEVANCE_FORMAT_ES = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
¿Quién es Quoc Le?

RESPONSE:
Después de completar su doctorado, Quoc Le se unió a Google Brain, donde ha estado \
trabajando en una variedad de proyectos de aprendizaje profundo. Quoc es muy respetado por muchos \
de sus pares, como Geoffrey Hinton, profesor adjunto en la \
Universidad de Montreal e imparte cursos sobre aprendizaje profundo.

STATEMENT:
Geoffrey Hinton está en la Universidad de Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. The phrase "Quoc es muy respetado por muchos de sus compañeros, como \
Geoffrey Hinton" from the RESPONSE shows that the relationship between Quoc \
Le and Geoffrey Hinton is that they are peers. For this reason, the subjects \
Quoc Le and Geoffrey Hinton are [{SYMBOL}].


Example 2:
QUESTION:
¿Quién es Quoc Le?

RESPONSE:
Después de completar su doctorado, Quoc Le se unió a Google Brain, donde ha estado trabajando en una variedad de proyectos de aprendizaje profundo. Geoffrey Hinton es profesor adjunto en la Universidad de Montreal, donde imparte cursos sobre aprendizaje profundo.

STATEMENT:
Geoffrey Hinton está en la Universidad de Montreal.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
Geoffrey Hinton. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and Geoffrey Hinton is. Thus, the subjects Quoc Le and \
Geoffrey Hinton are [{NOT_SYMBOL}].


Your Task:
QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_REVISE_FORMAT_ES = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "su", "ellas", "ella", "ellos")
- Unknown entities (e.g., "este evento", "la investigación", "la invención")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns es una empresa.

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. \
The company is headquartered in Irvine, California.

REVISED STATEMENT:
The subject in the statement "Acorns es una empresa." is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns es una empresa.
```


Example 2:
STATEMENT:
Él imparte un curso sobre aprendizaje profundo.

RESPONSE:
Después de completar su doctorado, Quoc Le se unió a Google Brain, \
donde ha estado trabajando en una variedad de proyectos de aprendizaje profundo. \
Le también es profesor adjunto en la Universidad de Montreal, \
donde imparte cursos sobre aprendizaje profundo.

REVISED STATEMENT:
The subject in the statement "Él imparte un curso sobre aprendizaje profundo" is "él". \
From the RESPONSE, we can see that this statement comes from the sentence "Le \
también es profesor adjunto en la Universidad de Montreal, \
donde imparte cursos sobre aprendizaje profundo.", meaning that "él" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le imparte cursos sobre aprendizaje profundo.
```


Example 3:
STATEMENT:
La serie de televisión se llama "You're the Worst".

RESPONSE:
Xóchitl Gómez comenzó su carrera como actriz en producciones \
teatrales e hizo su debut televisivo en 2016 con una aparición \
especial en la serie de Disney Channel "Raven's Home". \
También ha aparecido en las series de televisión "You're the Worst" y "Gentefied".

REVISED STATEMENT:
The subject of the statement "La serie de televisión se llama "You're the \
Worst"" is "la serie de televisión". This is a reference to an unknown entity, \
since it is unclear what television series is "la serie de televisión". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "la serie de televisión" is a \
vague reference that should be replaced by "la serie de televisión en \
la que apareció Xóchitl Gómez". Thus, the revised response is:
```
La serie de televisión en la que apareció Xóchitl Gómez se llama "You're the Worst".
```


Example 4:
STATEMENT:
Dean se unió a Google.

RESPONSE:
Jeff Dean es miembro senior de Google y director de Google AI, y lidera \
la investigación y el desarrollo en inteligencia artificial. \
Dean se unió a Google en 1999 y ha sido esencial para su desarrollo \
continuo en este campo.

REVISED STATEMENT:
The subject of the statement "Dean se unió a Google" is "Dean". From the \
response, we can see that "Dean" is the last name of "Jeff Dean". Therefore \
"Dean" is a non-full name, making it a vague reference. It should be replaced \
by "Jeff Dean", which is the full name. Thus, the revised response is:
```
Jeff Dean se unió a Google.
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""
#############
_RELEVANCE_FORMAT_AR = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
من هو كووك لو؟

RESPONSE:
بعد حصوله على درجة الدكتوراه، انضم Quoc Le إلى Google Brain، حيث كان يعمل على مجموعة متنوعة من مشاريع التعلم العميق. يحظى كووك باحترام كبير من قبل العديد من أقرانه، مثل جيفري هينتون، وهو أستاذ مساعد في جامعة مونتريال ويقوم بتدريس دورات حول التعلم العميق.

STATEMENT:
جيفري هينتون في جامعة مونتريال.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
جيفري هينتون. The phrase "يحظى Quoc باحترام كبير من قبل العديد من أقرانه، مثل جيفري هينتون" \
from the RESPONSE shows that the relationship between Quoc \
Le and جيفري هينتون is that they are peers. For this reason, the subjects \
Quoc Le and جيفري هينتون are [{SYMBOL}].


Example 2:
QUESTION:
من هو كووك لو؟

RESPONSE:
بعد حصوله على درجة الدكتوراه، انضم Quoc Le إلى Google Brain، حيث كان يعمل على مجموعة متنوعة من مشاريع التعلم العميق. جيفري هينتون هو أستاذ مساعد في جامعة مونتريال، حيث يقوم بتدريس دورات حول التعلم العميق.

STATEMENT:
جيفري هينتون في جامعة مونتريال.

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
جيفري هينتون. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and جيفري هينتون is. Thus, the subjects Quoc Le and \
جيفري هينتون are [{NOT_SYMBOL}].


Your Task:
QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_REVISE_FORMAT_AR = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "له"، "هم"، "لها")
- Unknown entities (e.g., "هذا الحدث"، "البحث"، "الاختراع")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
الجوز هي شركة.

RESPONSE:
الجوز هي شركة تكنولوجيا مالية تأسست في عام 2012 على يد والتر كروتندن، وجيف كروتندن، ومارك درو، وتقدم خدمات الاستثمار الصغير. يقع المقر الرئيسي للشركة في إرفاين، كاليفورنيا.

REVISED STATEMENT:
The subject in the statement "الجوز هي شركة" is "الجوز". "الجوز" is \
not a pronoun and does not reference an unknown entity. Furthermore, "الجوز" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "الجوز" is not a vague reference. Thus, the revised statement \
is:
```
الجوز هي شركة.
```


Example 2:
STATEMENT:
يقوم بتدريس دورات في التعلم العميق.

RESPONSE:
بعد حصوله على درجة الدكتوراه، انضم Quoc Le إلى Google Brain، حيث كان يعمل على مجموعة متنوعة من مشاريع التعلم العميق. لو هو أيضًا أستاذ مساعد في جامعة مونتريال، حيث يقوم بتدريس دورات حول التعلم العميق.

REVISED STATEMENT:
The subject in the statement "إنه يقوم بتدريس دورة عن التعلم العميق" is "هو". \
From the RESPONSE, we can see that this statement comes from the sentence "لو هو أيضًا أستاذ مساعد في جامعة مونتريال، حيث يقوم بتدريس دورات حول التعلم العميق.", meaning that "هو" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
يقوم Quoc Le بتدريس دورات حول التعلم العميق.
```


Example 3:
STATEMENT:
المسلسل التلفزيوني يسمى "أنت الأسوأ".

RESPONSE:
بدأت Xochitl Gomez مسيرتها التمثيلية في الإنتاج المسرحي، وظهرت لأول مرة على التلفزيون في عام 2016 مع ظهورها كضيفة في مسلسل قناة ديزني "Raven's Home". كما ظهرت في المسلسل التلفزيوني "أنت الأسوأ" و"جينتيفيد".

REVISED STATEMENT:
The subject of the statement "المسلسل التلفزيوني اسمه "أنت الأسوأ"" is "المسلسل التلفزيوني". This is a reference to an unknown entity, \
since it is unclear what television series is "المسلسل التلفزيوني". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "المسلسل التلفزيوني" is a \
vague reference that should be replaced by "المسلسل التلفزيوني الذي ظهر فيه Xochitl Gomez". Thus, the revised response is:
```
المسلسل التلفزيوني الذي ظهر فيه Xochitl Gomez يسمى "أنت الأسوأ".
```


Example 4:
STATEMENT:
انضم دين إلى جوجل.

RESPONSE:
جيف دين هو أحد كبار زملاء Google ورئيس قسم الذكاء الاصطناعي في Google، وهو يقود البحث والتطوير في مجال الذكاء الاصطناعي. انضم دين إلى شركة Google في عام 1999 وكان له دور أساسي في تطويرها المستمر في هذا المجال.

REVISED STATEMENT:
The subject of the statement "انضم دين إلى جوجل" is "عميد". From the \
response, we can see that "عميد" is the last name of "جيف دين". Therefore \
"عميد" is a non-full name, making it a vague reference. It should be replaced \
by "جيف دين", which is the full name. Thus, the revised response is:
```
انضم جيف دين إلى جوجل.
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""
#############
_RELEVANCE_FORMAT_BN = f"""\
In a given RESPONSE, two subjects are considered "{SYMBOL}" if the RESPONSE \
contains information that explains how the two subjects are related.


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of \
the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the \
QUESTION should be considered {SYMBOL}, based on the given definition of \
"{SYMBOL}."
4. Before showing your answer, think step-by-step and show your specific \
reasoning.
5. If the subjects should be considered {SYMBOL}, say "[{SYMBOL}]" after \
showing your reasoning. Otherwise show "[{NOT_SYMBOL}]" after showing your \
reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
QUESTION:
Quoc Le কে?

RESPONSE:
পিএইচডি শেষ করার পর, Quoc Le Google Brain-এ যোগ দেন, যেখানে তিনি বিভিন্ন গভীর শিক্ষার প্রকল্পে কাজ করছেন। Quoc তার অনেক সমবয়সীদের দ্বারা সুসম্মানিত, যেমন জিওফ্রে হিন্টন, যিনি মন্ট্রিল বিশ্ববিদ্যালয়ের একজন সহযোগী অধ্যাপক এবং গভীর শিক্ষার উপর কোর্স পড়ান।

STATEMENT:
জিওফ্রে হিন্টন মন্ট্রিল বিশ্ববিদ্যালয়ে আছেন।

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
জিওফ্রে হিন্টন. The phrase "জিওফ্রে হিন্টনের মতো তার সমবয়সীদের অনেকের কাছে Quoc সম্মানিত" from the RESPONSE shows that the relationship between Quoc \
Le and জিওফ্রে হিন্টন is that they are peers. For this reason, the subjects \
Quoc Le and জিওফ্রে হিন্টন are [{SYMBOL}].


Example 2:
QUESTION:
Quoc Le কে?

RESPONSE:
পিএইচডি শেষ করার পর, Quoc Le Google Brain-এ যোগ দেন, যেখানে তিনি বিভিন্ন গভীর শিক্ষার প্রকল্পে কাজ করছেন। জিওফ্রে হিন্টন হলেন মন্ট্রিল বিশ্ববিদ্যালয়ের একজন সহযোগী অধ্যাপক, যেখানে তিনি গভীর শিক্ষার কোর্স শেখান।

STATEMENT:
জিওফ্রে হিন্টন মন্ট্রিল বিশ্ববিদ্যালয়ে আছেন।

SOLUTION:
The subject of the QUESTION is Quoc Le. The subject of the STATEMENT is \
জিওফ্রে হিন্টন. While both subjects seem to be related to deep learning, \
the RESPONSE does not contain any phrases that explain what the relationship \
between Quoc Le and জিওফ্রে হিন্টন is. Thus, the subjects Quoc Le and \
জিওফ্রে হিন্টন are [{NOT_SYMBOL}].


Your Task:
QUESTION:
{_PROMPT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_REVISE_FORMAT_BN = f"""\
Vague references include but are not limited to:
- Pronouns (e.g., "তার", "তারা", "তার")
- Unknown entities (e.g., "এই ঘটনা", "গবেষণা", "উদ্ভাবন")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)


Instructions:
1. The following STATEMENT has been extracted from the broader context of the \
given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities \
from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. \
For example, given the response "Titanic is a movie starring Leonardo \
DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your \
reasoning. As part of your reasoning, be sure to identify the subjects in the \
STATEMENT and determine whether they are vague references. If they are vague \
references, identify the proper entity that they are referring to and be sure \
to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in \
a markdown code block.
7. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". \
Some examples have been provided for you to learn how to do this task.


Example 1:
STATEMENT:
Acorns একটি কোম্পানি.

RESPONSE:
Acorns হল একটি আর্থিক প্রযুক্তি কোম্পানি যা 2012 সালে Walter Cruttenden, Jeff Cruttenden, এবং Mark Dru দ্বারা প্রতিষ্ঠিত হয়েছিল যা মাইক্রো-বিনিয়োগ পরিষেবা প্রদান করে। কোম্পানির সদর দফতর ক্যালিফোর্নিয়ার আরভিনে অবস্থিত।

REVISED STATEMENT:
The subject in the statement "Acorns একটি কোম্পানি" is "Acorns". "Acorns" is \
not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" \
is not further specified in the RESPONSE, so we can assume that it is a full \
name. Therefore "Acorns" is not a vague reference. Thus, the revised statement \
is:
```
Acorns একটি কোম্পানি
```


Example 2:
STATEMENT:
তিনি গভীর শিক্ষার কোর্স শেখান।

RESPONSE:
পিএইচডি শেষ করার পর, Quoc Le Google Brain-এ যোগ দেন, যেখানে তিনি বিভিন্ন গভীর শিক্ষার প্রকল্পে কাজ করছেন। লে মন্ট্রিল বিশ্ববিদ্যালয়ের একজন সহযোগী অধ্যাপক, যেখানে তিনি গভীর শিক্ষার কোর্স শেখান।

REVISED STATEMENT:
The subject in the statement "তিনি গভীর শিক্ষার কোর্স শেখান।" is "তিনি". \
From the RESPONSE, we can see that this statement comes from the sentence "লে মন্ট্রিল বিশ্ববিদ্যালয়ের একজন সহযোগী অধ্যাপক, যেখানে তিনি গভীর শিক্ষার কোর্স শেখান।", meaning that "he" refers to "Le". From the \
RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a \
non-full name that should be replaced by "Quoc Le." Thus, the revised response \
is:
```
Quoc Le গভীর শিক্ষার কোর্স শেখায়।
```


Example 3:
STATEMENT:
টেলিভিশন সিরিজটির নাম "তুমি সবচেয়ে খারাপ"

RESPONSE:
Xochitl Gomez থিয়েটার প্রোডাকশনে তার অভিনয় জীবন শুরু করেন এবং তিনি 2016 সালে ডিজনি চ্যানেল সিরিজ "Raven's Home" এ অতিথি চরিত্রে অভিনয়ের মাধ্যমে তার টেলিভিশনে আত্মপ্রকাশ করেন। তিনি টেলিভিশন সিরিজ "ইউ আর দ্য ওয়ার্স্ট" এবং "জেন্টিফাইড" এও উপস্থিত হয়েছেন।

REVISED STATEMENT:
The subject of the statement "টেলিভিশন সিরিজটির নাম "তুমি সবচেয়ে খারাপ।"" is "টেলিভিশন সিরিজ". This is a reference to an unknown entity, \
since it is unclear what television series is "টেলিভিশন সিরিজ". From \
the RESPONSE, we can see that the STATEMENT is referring to the television \
series that Xochitl Gomez appeared in. Thus, "টেলিভিশন সিরিজ" is a \
vague reference that should be replaced by "Xochitl Gomez যে টেলিভিশন সিরিজে উপস্থিত হয়েছিল". Thus, the revised response is:
```
Xochitl গোমেজ যে টেলিভিশন সিরিজে হাজির হয়েছিল তার নাম "ইউ আর দ্য ওয়ার্স্ট"
```


Example 4:
STATEMENT:
ডিন গুগলে যোগ দেন।

RESPONSE:
জেফ ডিন হলেন একজন Google সিনিয়র ফেলো এবং Google AI-এর প্রধান, কৃত্রিম বুদ্ধিমত্তায় গবেষণা ও উন্নয়নে নেতৃত্ব দিচ্ছেন। ডিন 1999 সালে Google-এ যোগদান করেন এবং এই ক্ষেত্রে এটির ক্রমাগত উন্নয়নের জন্য অপরিহার্য।

REVISED STATEMENT:
The subject of the statement "ডিন গুগলে যোগ দেন" is "যাজক". From the \
response, we can see that "যাজক" is the last name of "জেফ ডিন". Therefore \
"যাজক" is a non-full name, making it a vague reference. It should be replaced \
by "জেফ ডিন", which is the full name. Thus, the revised response is:
```
জেফ ডিন গুগলে যোগ দিয়েছেন।
```


Your Task:
STATEMENT:
{_STATEMENT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}
"""



#############
_RELEVANCE_FORMAT = _RELEVANCE_FORMAT_EN
_REVISE_FORMAT = _REVISE_FORMAT_EN



def check_relevance(
    prompt: str,
    response: str,
    atomic_fact: str,
    model: modeling.Model,
    do_debug: bool = safe_config.debug_safe,
    max_retries: int = safe_config.max_retries,
) -> tuple[str, bool]:
  """Check if the atomic fact is relevant for answering the prompt."""
  full_prompt = _RELEVANCE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
  full_prompt = full_prompt.replace(_PROMPT_PLACEHOLDER, prompt)
  full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
  full_prompt = utils.strip_string(full_prompt)
  model_response, answer, num_tries = '', '', 0

  while not answer and num_tries <= max_retries:
    model_response = model.generate(full_prompt, do_debug=do_debug)
    answer = utils.extract_first_square_brackets(model_response)
    answer = answer if answer in [SYMBOL, NOT_SYMBOL] else None
    num_tries += 1

  answer = not answer or answer.lower() == SYMBOL.lower()
  return model_response, answer  # if no parsed answer, assume relevant


def revise_fact(
    response: str,
    atomic_fact: str,
    model: modeling.Model,
    do_debug: bool = safe_config.debug_safe,
    max_retries: int = safe_config.max_retries,
) -> tuple[str, str]:
  """Modify the atomic fact to be self-contained."""
  full_prompt = _REVISE_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
  full_prompt = full_prompt.replace(_RESPONSE_PLACEHOLDER, response)
  full_prompt = utils.strip_string(full_prompt)
  model_response, revised_fact, num_tries = '', '', 0

  while not revised_fact and num_tries <= max_retries:
    model_response = model.generate(full_prompt, do_debug=do_debug)
    revised_fact = utils.extract_first_code_block(
        model_response, ignore_language=True
    )
    num_tries += 1

  return model_response, revised_fact or atomic_fact


def main(
    prompt: str, response: str, atomic_fact: str, model: modeling.Model
) -> tuple[bool, str, dict[str, Any]]:
  """Check if the fact is relevant and modify it to be self-contained."""
  model_responses = {'atomic_fact': atomic_fact}
  model_responses['revised_fact'], atomic_fact = revise_fact(
      response=response, atomic_fact=atomic_fact, model=model
  )

  ######## HERE
  # model_responses['is_relevant'], is_relevant = check_relevance(
  #     prompt=prompt, response=response, atomic_fact=atomic_fact, model=model
  # )
  model_responses['is_relevant'] = "passing this"
  is_relevant = True
  return is_relevant, atomic_fact, model_responses
