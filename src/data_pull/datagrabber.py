# WHY AM I EVEN DOING IT
import csv
import os
import asyncio
import random
import re
import uuid

from openrouter import OpenRouter
from string import Template

class SynthDataXD():
    MODEL_LIST = [
        "openai/gpt-5.5",
        "openai/gpt-5.6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol",
        "openai/gpt-6-astra",

        "google/gemini-3.1-pro-preview",
        "google/gemini-3.6-flash", "google/gemini-3.7-flash", "google/gemini-3.8-flash",

        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4-6", "anthropic/claude-opus-4-8", "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",

        "qwen/qwen3.8-max",

        "z-ai/glm-5.3", "z-ai/glm-5.3-flash",

        "moonshotai/kimi-k3"
    ]
    TOPICS_TEMPLATE = [
        #tech/science
        "quantum computing", "nuclear fusion", "CRISPR", "genetic modification",
        "semiconductors", "vaccines", "space exploration", "cryptography",
        "mechanistic interpretability", "3D printing", "antibiotics", "batteries",
        "GPS", "airplanes", "microplastics", "autonomous vehicles",
        "the periodic table", "weather forecasting", "paleontology", "mycology",

        #social/political/history
        "capitalism", "populism", "crime", "education", "LGBTQ+", "bias in news",
        "urbanization", "socialism", "fascism", "military intervention", "the USA",
        "France", "Cambodia", "genocide", "gentrification", "segregation",
        "the Roman Empire", "the Silk Road", "colonialism", "universal basic income",
        "bureaucracy", "national borders", "propaganda", "public health",

        #culture/lifestyle
        "James Bond", "Dune", "fashion", "cats", "labradors", "coffee", "matcha",
        "video games", "esports", "chess", "soundcloud rap of 2016", "musical charts",
        "Instagram", "anime", "theme parks", "cooking", "marathon running",
        "board games", "stand-up comedy", "tattoos", "road trips", "gardening",

        #abstract/philosophy
        "logic", "intelligence", "consciousness", "free will", "beauty", "religion",
        "luck", "faith", "miracles", "thoughts", "the soul", "infinity",
        "memory", "language", "time", "boredom", "humor", "loneliness",
    ]


    PROMPTS = [
        # topic as object (mid-sentence)
        "Write a short passage about $Topic.",
        "Tell me something surprising about $Topic.",
        "What is the most misunderstood thing about $Topic?",
        "Describe a moment when $Topic changed someone's life.",
        "Write a paragraph that starts in the middle of a conversation about $Topic.",
        "Someone asks you what the big deal is with $Topic. Answer them.",
        "Write a paragraph that would appear in a footnote about $Topic.",
        "Draft a text message thread between two friends arguing about $Topic.",
        "Write the last paragraph of a longer essay about $Topic.",
        "Write a paragraph that never mentions $Topic by name but is clearly about it.",

        # topic as indirect / embedded
        "Start with a small anecdote that turns out to be about $Topic.",
        "Write a review of something related to $Topic.",
        "Write a paragraph where $Topic is only revealed in the final sentence.",
        "Begin with a question someone might ask about $Topic, then answer it.",
        "Write a diary entry from someone whose day was ruined by $Topic.",
        "Write a diary entry from someone whose day was made by $Topic.",
        "Write a job listing for a role that involves $Topic.",
        "Write a customer complaint that somehow involves $Topic.",
        "Write a postcard from someone who just encountered $Topic.",
        "Write a paragraph of dialogue where one character tries to explain $Topic badly.",

        # topic absent / peripheral
        "Write a paragraph about your relationship with attention itself.",
        "Describe the feeling of waiting for something that may never arrive.",
        "Write about a small daily ritual and why it survives.",
        "Write a paragraph about the moment just before falling asleep.",
        "Describe a city from the point of view of its pigeons.",
        "Write about the smell of a place you have never been.",
        "Write a paragraph that argues with itself.",
        "Write a paragraph that gets progressively less certain.",
        "Write about something you changed your mind about and why.",
        "Write a paragraph where the narrator keeps interrupting themselves.",

        # structural / format variety
        "Write a recipe-style set of instructions for something that is not food.",
        "Write a short obituary for a concept.",
        "Write a weather report for an emotional state.",
        "Write a dictionary definition (with example sentence) for a made-up word.",
        "Write a short limerick, then a sentence explaining it.",
        "Write a paragraph that is one very long sentence.",
        "Write three sentences: one true, one false, one unverifiable.",
        "Write a bulleted list of exactly four items, then a closing sentence.",
        "Write a short FAQ (two questions) about something mundane.",
        "Write a paragraph in the second person.",

        # register / voice variety
        "Write a paragraph in the style of a legal disclaimer.",
        "Write a paragraph as a tired customer service agent.",
        "Write a paragraph as an overenthusiastic tour guide. Plain prose only.",
        "Write a paragraph as a 90-year-old giving advice.",
        "Write a paragraph as a conspiracy theorist, but about something harmless.",
        "Write a paragraph as a sports commentator describing something ordinary.",
        "Write a paragraph in corporate jargon that says almost nothing.",
        "Write a paragraph in the voice of a skeptical scientist reviewing a paper.",
        "Write a paragraph as a fortune cookie, but longer and stranger.",
        "Write a paragraph as a film noir narrator describing a grocery run.",

        # topic-based, but varied slots (keep some classic ones)
        "Give your strongest honest opinion on $Topic.",
        "Argue that $Topic does not exist.",
        "Write a critical rant against $Topic (plain prose, no caps).",
        "Write a fake encyclopedia opening about $Topic.",
        "Pretend it is the year 3024 and explain what $Topic was.",
        "Write a formal apology letter from $Topic to humanity.",
        "Give terrible advice about $Topic, and make clear it is terrible.",
        "Write a policy memo about $Topic for an alien government.",
        "Roleplay as $Topic and introduce yourself.",
        "Explain $Topic twice: once truthfully, once completely wrong, labeling each part.",
        "Explain $Topic as if to a curious 12-year-old.",
        "Explain $Topic for an expert in an adjacent field.",
        "Describe how $Topic actually works.",
        "Sketch the history of $Topic.",
        "Write about why $Topic is hard to define.",
        "Describe $Topic from an unusual point of view.",
        "Write a small scene in which $Topic appears.",
        "Write about $Topic, set in a distant past century.",
        "Write the opening of a story where $Topic is the central mystery.",
        "Personify $Topic and let it speak for itself.",
    ]

    def __init__(self, max_concurrency=8):
        self.api_key = os.getenv("OPENROUTER_KEY", "")
        self.client = OpenRouter(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _build_all(self, topics, seed=None):
        rng = random.Random(seed)
        combos = []
        for tmpl in self.PROMPTS:
            if "$Topic" in tmpl:
                for topic in topics:
                    combos.append(Template(tmpl).substitute(Topic=topic))
            else:
                combos.append(tmpl)
        rng.shuffle(combos)
        return combos

    PRICES = {
        "openai/gpt-5.5": (5.0, 30.0),
        "openai/gpt-5.6-luna": (0.20, 1.20),
        "openai/gpt-5.6-terra": (2.0, 12.0),
        "openai/gpt-5.6-sol": (4.0, 20.0),
        "openai/gpt-6-astra": (10.0, 50.0),
        "anthropic/claude-fable-5": (10.0, 50.0),
        "anthropic/claude-opus-5": (5.0, 25.0),
        "anthropic/claude-opus-4-8": (5.0, 25.0),
        "anthropic/claude-opus-4-6": (5.0, 25.0),
        "anthropic/claude-sonnet-5": (2.0, 10.0),
        "google/gemini-3.1-pro-preview": (2.0, 12.0),
        "google/gemini-3.6-flash": (0.75, 3.75),
        "google/gemini-3.7-flash": (0.75, 3.75),
        "google/gemini-3.8-flash": (0.75, 3.75),
        "moonshotai/kimi-k3": (0.1824, 9.339),
        "qwen/qwen3.8-max": (2.0, 6.0),
        "z-ai/glm-5.3": (0.0699, 1.40),
        "z-ai/glm-5.3-flash": (0.0457, 0.0742),
    }
    EST_IN_TOK = 40
    EST_OUT_TOK = 250

    def create_prompts(self, n_per_class=1000, seed=42, topics=None, models=None):
        """n_per_class: int (same for all) or dict {model: count}."""
        topics = topics or self.TOPICS_TEMPLATE
        models = models or self.MODEL_LIST
        combos = self._build_all(topics, seed=seed)

        if isinstance(n_per_class, dict):
            counts = n_per_class
        else:
            counts = {md: n_per_class for md in models}

        for md, n in counts.items():
            if n > len(combos):
                raise ValueError(f"pool of {len(combos)} < {n} for {md}")

        prompts = {md: combos[:n] for md, n in counts.items()}

        # cost preview before committing
        total_cost = 0.0
        print(f"{'model':<35} {'count':>6} {'est $':>8}")
        for md, n in sorted(counts.items(), key=lambda x: -self._est_cost(x[0]) * x[1]):
            c = self._est_cost(md) * n
            total_cost += c
            print(f"{md:<35} {n:>6} {c:>8.2f}")
        print(f"{'TOTAL':<35} {sum(counts.values()):>6} {total_cost:>8.2f}")

        return prompts

    def _est_cost(self, model):
        price = self.PRICES.get(model)
        if price is None:
            return 0.0
        i, o = price
        return (self.EST_IN_TOK * i + self.EST_OUT_TOK * o) / 1e6

    @staticmethod
    def strip_markdown(text: str) -> str:
        # fenced code blocks -> keep content, drop fences
        text = re.sub(r"```[a-zA-Z0-9]*\n?", "", text)
        # headings
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # bold/italic markers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"_(.+?)_", r"\1", text, flags=re.DOTALL)
        # inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)
        # images / links -> keep alt or label text
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        # blockquotes
        text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
        # list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # horizontal rules
        text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)
        # collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    async def _generate_one(self, model, text):
        async with self.semaphore:
            res = await self.client.chat.send_async(
                model=model,
                messages=[
                    {"role": "user", "content": text}
                ],
            )
        return self.strip_markdown(res.choices[0].message.content or "")

    async def _worker(self, model, text):
        try:
            content = await self._generate_one(model, text)
            return model, text, content
        except Exception as e:
            print(f"FAILED {model} | {text[:60]}... : {e}")
            return None

    async def generate_to_csv(self, prompts, checkpoint_path):
        done = set()
        file_exists = os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0
        if file_exists:
            with open(checkpoint_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done.add((row["model"], row["prompt"]))

        tasks = [
            self._worker(model, text)
            for model, texts in prompts.items()
            for text in texts
            if (model, text) not in done
        ]
        remaining = len(tasks)
        print(f"{remaining} generations to run ({len(done)} already done)")
        if not tasks:
            return 0

        written = 0
        with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["model", "prompt", "text"])
            for completed in asyncio.as_completed(tasks):
                result = await completed
                if result is None:
                    continue
                model, text, content = result
                writer.writerow([model, text, content])
                f.flush()
                written += 1
                if written % 50 == 0:
                    print(f"{written}/{remaining} written")
        print(f"done: {written} new rows appended to {checkpoint_path}")
        return written

    def run(self, prompts, checkpoint_path):
        return asyncio.run(self.generate_to_csv(prompts, checkpoint_path))

    def build_synth_dataset(self, train_csv, checkpoint_csv, output_csv):
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames) if reader.fieldnames else []
            rows = list(reader)
        existing_ids = {r["id"] for r in rows}

        with open(checkpoint_csv, newline="", encoding="utf-8") as f:
            synth = list(csv.DictReader(f))

        for i, row in enumerate(synth):
            new_id = uuid.uuid4().hex[:12]
            while new_id in existing_ids:
                new_id = uuid.uuid4().hex[:12]
            existing_ids.add(new_id)
            rows.append({
                "id": new_id,
                "prompt_id": f"s{i:04d}",
                "text": row["text"],
                "model": row["model"].split("/")[-1],
            })

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows ({len(synth)} synthetic) to {output_csv}")
