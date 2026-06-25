"""Bilingual BZU dataset generator.

Creates one large synthetic dataset of BZU (Birzeit University) student and
community content in **Palestinian-dialect Arabic** and **English**, spanning
three categories:

    1. Course Feedback
    2. Student Decisions  (student movements & university decisions)
    3. University Discussions

Each record has the fields:

    id, language, category, course_name, professor_name, text, sentiment, date

Text is produced by a tiny template engine: every (category, language,
sentiment) combination has many templates, and each template contains
``{placeholders}`` that are filled from large synonym pools. Combined with
random courses/professors and a light "style" decorator (emojis, hashtags,
casing), this yields thousands of diverse, mostly-unique comments.

No machine-learning models are built here — this only generates the dataset.

Usage:
    python src/data/generate_dataset.py                 # ~4000 records
    python src/data/generate_dataset.py --n 5000 --seed 7
    python src/data/generate_dataset.py --out data/raw/bzu_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CATEGORIES = ["Course Feedback", "Student Decisions", "University Discussions"]
SENTIMENTS = ["positive", "negative", "neutral"]

DEFAULT_N = 4000
DEFAULT_OUT = Path("data/raw/bzu_dataset.csv")
DATE_START = date(2022, 9, 1)
DATE_END = date(2026, 6, 15)

FIELDNAMES = [
    "id", "language", "category", "course_name",
    "professor_name", "text", "sentiment", "date",
]


# --------------------------------------------------------------------------- #
# Courses & professors (BZU-realistic)
# --------------------------------------------------------------------------- #
COURSES = [
    {"en": "Algorithms", "ar": "خوارزميات"},
    {"en": "Data Structures", "ar": "تراكيب بيانات"},
    {"en": "Operating Systems", "ar": "أنظمة تشغيل"},
    {"en": "Natural Language Processing", "ar": "معالجة اللغات الطبيعية"},
    {"en": "Database Systems", "ar": "قواعد بيانات"},
    {"en": "Computer Networks", "ar": "شبكات حاسوب"},
    {"en": "Linear Algebra", "ar": "جبر خطي"},
    {"en": "Calculus", "ar": "تفاضل وتكامل"},
    {"en": "Physics", "ar": "فيزياء"},
    {"en": "Software Engineering", "ar": "هندسة برمجيات"},
    {"en": "Artificial Intelligence", "ar": "ذكاء اصطناعي"},
    {"en": "Digital Logic", "ar": "دارات منطقية"},
    {"en": "Microprocessors", "ar": "معالجات دقيقة"},
    {"en": "Probability and Statistics", "ar": "احتمالات وإحصاء"},
    {"en": "Discrete Mathematics", "ar": "رياضيات متقطعة"},
    {"en": "Object Oriented Programming", "ar": "برمجة كائنية"},
    {"en": "Web Development", "ar": "تطوير ويب"},
    {"en": "Embedded Systems", "ar": "أنظمة مدمجة"},
    {"en": "Machine Learning", "ar": "تعلم آلي"},
    {"en": "Compilers", "ar": "مترجمات"},
    {"en": "Computer Architecture", "ar": "معمارية حاسوب"},
    {"en": "Arabic Language", "ar": "لغة عربية"},
    {"en": "English Language", "ar": "لغة إنجليزية"},
    {"en": "Islamic Culture", "ar": "ثقافة إسلامية"},
]

PROFESSORS = [
    {"en": "Ahmad Saleh", "ar": "أحمد صالح"},
    {"en": "Mahmoud Khalil", "ar": "محمود خليل"},
    {"en": "Rana Odeh", "ar": "رنا عودة"},
    {"en": "Yousef Hamdan", "ar": "يوسف حمدان"},
    {"en": "Lina Barghouthi", "ar": "لينا البرغوثي"},
    {"en": "Khaled Nazzal", "ar": "خالد نزال"},
    {"en": "Sami Abu Zahra", "ar": "سامي أبو زهرة"},
    {"en": "Huda Shaheen", "ar": "هدى شاهين"},
    {"en": "Tariq Awad", "ar": "طارق عوض"},
    {"en": "Nidal Qasem", "ar": "نضال قاسم"},
    {"en": "Maha Daraghmeh", "ar": "مها دراغمة"},
    {"en": "Omar Zaid", "ar": "عمر زيد"},
    {"en": "Fadi Masri", "ar": "فادي المصري"},
    {"en": "Reem Salah", "ar": "ريم صلاح"},
    {"en": "Bashar Hijazi", "ar": "بشار حجازي"},
    {"en": "Areej Nakhleh", "ar": "أريج نخلة"},
]


# --------------------------------------------------------------------------- #
# Placeholder pools
# --------------------------------------------------------------------------- #
POOLS: dict[str, dict[str, list[str]]] = {
    "en": {
        "intens": ["really", "very", "so", "honestly", "pretty", "super", "quite"],
        "pos_adv": ["well", "clearly", "in a simple way", "step by step"],
        "pos_adj": ["clear", "engaging", "well-organized", "helpful", "practical",
                    "interesting", "well-structured", "useful"],
        "pos_adj2": ["supportive", "knowledgeable", "friendly", "patient",
                     "inspiring", "approachable", "fair"],
        "neg_adj": ["confusing", "disorganized", "boring", "stressful",
                    "overwhelming", "unfair", "exhausting", "outdated", "messy"],
        "exam_word": ["exam", "midterm", "final", "quiz"],
        "decision": ["a full strike", "online learning", "in-person classes",
                     "a one-week class suspension", "the new schedule changes",
                     "suspending classes"],
        "mode": ["online", "in-person", "hybrid"],
        "issue": ["the rising tuition fees", "the fee increase",
                  "the new exam policy", "the poor conditions",
                  "the schedule changes", "the class suspensions",
                  "the lack of transparency"],
        "topic": ["registration", "tuition fees", "the bus transportation",
                  "the dorms", "the library", "graduation procedures",
                  "the student clubs", "campus events", "the facilities",
                  "the campus internet", "parking"],
    },
    "ar": {
        "intens": ["كتير", "جداً", "عنجد", "والله", "بصراحة"],
        "pos_adv": ["منيح", "بطريقة حلوة", "بشكل بسيط", "خطوة بخطوة"],
        "pos_adj": ["حلوة", "ممتعة", "مفيدة", "منظمة", "مشوقة", "سهلة الفهم", "رائعة"],
        "pos_adj2": ["متعاون", "فاهم", "محترم", "صبور", "ملهم", "قريب من الطلاب", "عادل"],
        "neg_adj": ["صعبة", "مملة", "معقدة", "متعبة", "ظالمة", "مرهقة", "قديمة", "فوضى"],
        "exam_word": ["الامتحان", "الميدتيرم", "الفاينل", "الكويز"],
        "decision": ["الإضراب الشامل", "التعليم الإلكتروني", "الدوام الوجاهي",
                     "تعليق الدوام أسبوع", "تغييرات الجدول", "تعليق المحاضرات"],
        "mode": ["اونلاين", "وجاهي", "مدمج"],
        "issue": ["رفع الأقساط", "زيادة الرسوم", "سياسة الامتحانات الجديدة",
                  "سوء الأوضاع", "تغييرات الجدول", "تعليق المحاضرات",
                  "غياب الشفافية"],
        "topic": ["التسجيل", "الأقساط", "مواصلات الباصات", "السكنات", "المكتبة",
                  "إجراءات التخرج", "النوادي الطلابية", "فعاليات الحرم",
                  "المرافق", "إنترنت الجامعة", "المواقف"],
    },
}


# --------------------------------------------------------------------------- #
# Templates: TEMPLATES[category][lang][sentiment] -> list[str]
# --------------------------------------------------------------------------- #
TEMPLATES: dict[str, dict[str, dict[str, list[str]]]] = {
    "Course Feedback": {
        "en": {
            "positive": [
                "Dr. {professor} explains {course} {intens} {pos_adv}; the lectures are {pos_adj} and easy to follow.",
                "Honestly {course} was one of the best courses I took — {professor} is {pos_adj2} and always ready to help.",
                "The labs in {course} are {pos_adj} and the projects actually teach you something useful.",
                "Grading in {course} is fair and Dr. {professor} gives clear feedback on every assignment.",
                "I loved the project in {course}; it was challenging but {pos_adj} and well guided.",
                "Attendance isn't stressful in {course} and {professor} makes every lecture worth it.",
                "{professor} is one of the most {pos_adj2} professors at BZU; {course} felt easy thanks to the great teaching.",
                "The {exam_word} in {course} was fair and matched what we studied — respect to Dr. {professor}.",
            ],
            "negative": [
                "{course} is {intens} {neg_adj}; the {exam_word} was way too hard and unfair.",
                "Dr. {professor} rushes through {course} and the lectures are {neg_adj}.",
                "The grading in {course} is {neg_adj}; you lose marks for no reason.",
                "Too many projects in {course} and not enough time — it's {neg_adj}.",
                "The labs in {course} are a mess and {professor} barely explains anything.",
                "Attendance is strict in {course} but the lectures aren't even worth showing up for.",
                "I regret taking {course}; the {exam_word}s are {neg_adj} and the material is outdated.",
                "{professor} makes {course} so {neg_adj}, half the class is lost by week three.",
            ],
            "neutral": [
                "{course} with Dr. {professor} is okay — nothing special but not bad either.",
                "The workload in {course} is average; a couple of projects and two {exam_word}s.",
                "{course} is a standard requirement; the lectures are fine and the grading is reasonable.",
                "Attendance in {course} is normal and the {exam_word} is about what you'd expect.",
                "Dr. {professor} teaches {course} in a typical way; some topics are interesting, others less so.",
                "Took {course} this semester; it's manageable if you keep up with the labs.",
            ],
        },
        "ar": {
            "positive": [
                "الدكتور {professor} بشرح {course} {intens} {pos_adv}، المحاضرات {pos_adj} والواحد بفهم بسرعة.",
                "بصراحة {course} من أحلى المواد اللي أخدتها، {professor} {pos_adj2} ودايماً مستعد يساعد.",
                "مختبرات {course} {pos_adj} والمشاريع بتعلمك إشي مفيد عنجد.",
                "العلامات في {course} عادلة والدكتور {professor} بيعطي ملاحظات واضحة على كل وظيفة.",
                "حبيت المشروع في {course}، كان صعب بس {pos_adj} والدكتور وجهنا منيح.",
                "{professor} دكتور {pos_adj2} وبشرح {course} بطريقة بتوصل، صارت سهلة علينا.",
                "{exam_word} تبع {course} كان عادل وعلى قد اللي درسناه، يسلمو دكتور {professor}.",
            ],
            "negative": [
                "{course} {intens} {neg_adj}، {exam_word} كان تعجيزي وظالم.",
                "الدكتور {professor} بيركض بالمادة و{course} محاضراتها {neg_adj}.",
                "تصحيح {course} {neg_adj}، بتخسر علامات من غير سبب.",
                "مشاريع {course} كتير والوقت مش كافي، الوضع {neg_adj}.",
                "مختبرات {course} فوضى و{professor} بالكاد بيشرح إشي.",
                "ندمت إني أخدت {course}، {exam_word} {neg_adj} والمادة قديمة.",
                "{professor} بيخلي {course} {neg_adj}، نص الشعبة ضايع من أول شهر.",
            ],
            "neutral": [
                "{course} مع الدكتور {professor} عادية، لا أحلى ولا أوحش.",
                "الشغل في {course} متوسط، كم مشروع وامتحانين.",
                "{course} مادة متطلب عادي، المحاضرات ماشية والعلامات معقولة.",
                "الحضور في {course} عادي و{exam_word} متوقع.",
                "الدكتور {professor} بيشرح {course} بطريقة عادية، في مواضيع مشوقة وفي أقل.",
                "أخدت {course} هالفصل، بتمشي حالها إذا ضليت متابع المختبرات.",
            ],
        },
    },

    "Student Decisions": {
        "en": {
            "positive": [
                "Great news — the university approved {decision}; students have been asking for this for a while.",
                "Glad the student council pushed for {decision}; it's a step in the right direction.",
                "Switching to {mode} learning was the right call and most students are relieved.",
                "Proud of the student movement today; our voice on {issue} is finally being heard.",
                "Respect to everyone who organized peacefully — {decision} shows that unity works.",
            ],
            "negative": [
                "Students declared a strike over {issue}; we want our rights and no one is listening.",
                "We're protesting {issue} — the administration keeps ignoring us.",
                "The sudden decision about {decision} messed up everyone's plans and no one was consulted.",
                "Another schedule change with no warning; students are furious about {issue}.",
                "Forcing {mode} classes on us mid-semester is unfair and disorganized.",
                "Suspending classes again? {issue} is turning the whole semester into chaos.",
            ],
            "neutral": [
                "The university announced {decision} starting next week.",
                "Classes will move to {mode} until further notice, according to the administration.",
                "The student union scheduled a meeting to discuss {issue}.",
                "A general assembly was called to vote on {issue}.",
                "There is a student gathering today regarding {decision}; details on the union page.",
            ],
        },
        "ar": {
            "positive": [
                "أخيراً الجامعة وافقت على {decision}، الطلاب من زمان عم يطالبو فيه.",
                "منيح إنه مجلس الطلبة ضغط لأجل {decision}، خطوة بالاتجاه الصح.",
                "التحول لـ {mode} كان قرار صح ومعظم الطلاب ارتاحو.",
                "فخور بالحراك الطلابي اليوم، صوتنا بخصوص {issue} أخيراً انسمع.",
                "احترام لكل مين نظّم بشكل سلمي، {decision} أثبت إنه الوحدة بتفرق.",
            ],
            "negative": [
                "الطلاب أعلنو الإضراب بسبب {issue}، بدنا حقوقنا وما حدا بسمعنا.",
                "احنا بنعتصم ضد {issue}، الإدارة كل مرة بتتجاهلنا.",
                "قرار {decision} الفجائي بوّظ خطط الكل وما حدا استشار الطلاب.",
                "كمان تغيير بالجدول بدون ما يخبرونا، الطلاب متضايقين من {issue}.",
                "فرض {mode} علينا بنص الفصل إشي مش عادل وغير منظم.",
                "كمان تعليق للمحاضرات؟ {issue} عم يحوّل الفصل كله لفوضى.",
            ],
            "neutral": [
                "الجامعة أعلنت {decision} ابتداءً من الأسبوع الجاي.",
                "حيتحول الدوام إلى {mode} لحد إشعار آخر حسب الإدارة.",
                "مجلس الطلبة حدد اجتماع لمناقشة {issue}.",
                "في دعوة لاجتماع عام للتصويت على {issue}.",
                "في تجمّع طلابي اليوم بخصوص {decision}، التفاصيل على صفحة المجلس.",
            ],
        },
    },

    "University Discussions": {
        "en": {
            "positive": [
                "The {topic} situation at BZU has improved a lot this semester.",
                "Honestly the {topic} on campus is {pos_adj} now; big improvement.",
                "Shoutout to whoever sorted out the {topic} — it's so much better.",
                "Finally the {topic} is working smoothly; makes student life way easier.",
            ],
            "negative": [
                "The {topic} at BZU is {neg_adj}; something really needs to change.",
                "Every semester the same problem with the {topic} — it's exhausting.",
                "Can we talk about how {neg_adj} the {topic} is? Students deserve better.",
                "I wasted half a day because of the {topic}; this is unacceptable.",
            ],
            "neutral": [
                "Does anyone have info about the {topic} for this semester?",
                "Quick question about the {topic} — where do I even start?",
                "They updated the {topic} policy; check the portal for details.",
                "Looking for advice on the {topic}, any tips from senior students?",
            ],
        },
        "ar": {
            "positive": [
                "وضع {topic} في الجامعة تحسّن كتير هالفصل، الله يعطيهم العافية.",
                "بصراحة {topic} صار أحسن بكتير، تطور واضح.",
                "شكراً لكل مين اهتم بـ {topic}، صار مرتب ومريح.",
                "أخيراً {topic} صار ماشي تمام، سهّل علينا حياتنا الجامعية.",
            ],
            "negative": [
                "{topic} في بيرزيت صار وجع راس، لازم إشي يتغير.",
                "كل فصل نفس المشكلة مع {topic}، تعبنا والله.",
                "مشكلة {topic} ما إلها حل، الطلاب بستاهلو أحسن.",
                "ضيّعت نص يوم بسبب {topic}، هاد إشي مش مقبول.",
            ],
            "neutral": [
                "حدا عندو معلومات عن {topic} لهالفصل؟",
                "سؤال بسيط عن {topic}، من وين أبدأ؟",
                "حدّثو إجراءات {topic}، شوفو البوابة للتفاصيل.",
                "بدور على نصيحة بخصوص {topic}، في حدا من الطلاب القدامى بساعد؟",
            ],
        },
    },
}


# --------------------------------------------------------------------------- #
# Style decorator (adds realistic variety)
# --------------------------------------------------------------------------- #
EMOJIS = {
    "positive": ["🔥", "👏", "🙏", "😍", "💯"],
    "negative": ["😡", "😞", "👎", "🤦", "😤"],
    "neutral": ["🤔", "📌", "❓", "🧐"],
}
HASHTAGS = {
    "en": ["#BZU", "#Birzeit", "#StudentLife", "#BZUstudents"],
    "ar": ["#بيرزيت", "#جامعة_بيرزيت", "#طلاب_بيرزيت", "#حراك_طلابي"],
}

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def render(template: str, lang: str, ctx: dict, rng: random.Random) -> str:
    """Fill ``{placeholders}`` from ``ctx`` first, else from POOLS[lang]."""
    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key in ctx:
            return ctx[key]
        options = POOLS[lang].get(key)
        return rng.choice(options) if options else match.group(0)

    return _PLACEHOLDER_RE.sub(repl, template)


def decorate(text: str, lang: str, sentiment: str, rng: random.Random) -> str:
    """Apply light, random stylistic variation for realism/diversity."""
    # Occasional casual lowercasing (English only).
    if lang == "en" and rng.random() < 0.15:
        text = text.lower()
    # Occasional extra punctuation emphasis.
    if rng.random() < 0.12:
        text = text.rstrip(".") + rng.choice(["!", "!!", "...", "؟" if lang == "ar" else "?"])
    # Occasional trailing emoji matching sentiment.
    if rng.random() < 0.30:
        text = f"{text} {rng.choice(EMOJIS[sentiment])}"
    # Occasional trailing hashtag.
    if rng.random() < 0.15:
        text = f"{text} {rng.choice(HASHTAGS[lang])}"
    return text


def random_date(rng: random.Random) -> str:
    span = (DATE_END - DATE_START).days
    return (DATE_START + timedelta(days=rng.randrange(span + 1))).isoformat()


# --------------------------------------------------------------------------- #
# Record generation
# --------------------------------------------------------------------------- #
def make_record(rid: int, rng: random.Random) -> dict:
    lang = rng.choice(["ar", "en"])
    category = rng.choices(CATEGORIES, weights=[0.40, 0.28, 0.32])[0]
    sentiment = rng.choices(SENTIMENTS, weights=[0.36, 0.40, 0.24])[0]

    ctx: dict = {}
    course_name = ""
    professor_name = ""
    if category == "Course Feedback":
        course = rng.choice(COURSES)
        professor = rng.choice(PROFESSORS)
        course_name = course[lang]
        professor_name = professor[lang]
        ctx = {"course": course_name, "professor": professor_name}

    template = rng.choice(TEMPLATES[category][lang][sentiment])
    text = render(template, lang, ctx, rng)
    text = decorate(text, lang, sentiment, rng)

    return {
        "id": rid,
        "language": lang,
        "category": category,
        "course_name": course_name,
        "professor_name": professor_name,
        "text": text,
        "sentiment": sentiment,
        "date": random_date(rng),
    }


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[str] = set()
    rid = 1
    attempts = 0
    max_attempts = n * 40

    while len(records) < n and attempts < max_attempts:
        attempts += 1
        rec = make_record(rid, rng)
        key = (rec["language"], rec["text"])
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)
        rid += 1

    return records


# --------------------------------------------------------------------------- #
# Output + reporting
# --------------------------------------------------------------------------- #
def write_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Arabic displays correctly when opened in Excel.
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict]) -> None:
    def counts(field: str) -> dict:
        out: dict = {}
        for r in records:
            out[r[field]] = out.get(r[field], 0) + 1
        return out

    print(f"Total records : {len(records)}")
    print(f"Unique texts  : {len({(r['language'], r['text']) for r in records})}")
    print(f"By language   : {counts('language')}")
    print(f"By category   : {counts('category')}")
    print(f"By sentiment  : {counts('sentiment')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the bilingual BZU dataset.")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"number of records (default {DEFAULT_N})")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output CSV path (default {DEFAULT_OUT})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.n <= 20000):
        raise SystemExit("--n must be between 1 and 20000")

    records = generate(args.n, args.seed)
    if len(records) < args.n:
        print(f"[warn] only produced {len(records)} unique records "
              f"(requested {args.n}); consider more templates/pools.")

    write_csv(records, args.out)
    print(f"Wrote {len(records)} records -> {args.out}")
    summarize(records)


if __name__ == "__main__":
    main()
