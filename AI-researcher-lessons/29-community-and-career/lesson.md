# Lesson 29 — Community & Career

---

## The problem

You can now do the work: find a question, run the experiment, write it up. But research
done in a vacuum barely counts. Nobody can build on results they never see; nobody can
correct errors they never see; and no one will hire, fund, or collaborate with a
researcher they have never heard of.

The uncomfortable truth and the liberating truth are the same sentence:

> **Research is a community sport, and the community is more open than any other elite
> profession on Earth.**

There is no bar exam, no license, no required degree. The field runs on public artifacts —
papers, repos, posts — and it genuinely cannot tell (and increasingly doesn't ask)
whether the person behind a great artifact has a PhD from Stanford or a laptop in a
bedroom. This lesson maps where the community lives, how beginners earn credibility,
and what the career paths honestly look like — **especially the paths that require no
university pedigree and no US/EU passport.**

---

## Intuition: the guild with an open door

Medieval guilds had a rule: to join, you presented a *masterpiece* — one piece of work
proving you could do the craft. ML research still works this way, except the guild hall
is public and the door is unlocked. Your masterpiece is a public artifact: a clean
reproduction, a sharp explainer, a well-run tiny experiment. People walk through the
door by *doing the work where others can see it*.

What does NOT work is the inverse: credentials without artifacts, or asking for
permission to start. Nobody grants permission. You post.

---

## Where researchers live online

```
PLACE            WHAT HAPPENS THERE                YOUR MOVE
──────────────────────────────────────────────────────────────────────
X/Twitter        paper announcements, debates,     follow researchers, share
                 hiring; the field's town square    your reproductions
arXiv            the actual literature             daily/weekly cs.CL, cs.LG
                 (lesson 13 taught you to read it)  skim; alerts on your topic
GitHub           the field's real currency:        publish everything you
                 code that runs                     build; contribute upstream
Hugging Face     models, datasets, spaces, and     host your models/demos;
                 an active community layer          write model cards
Discords         EleutherAI, Hugging Face, and     lurk → ask one good
                 paper-reading servers; where       question → help others
                 collaborations actually form
Blogs/Substack   long-form explainers and          publish yours (lesson 28
                 research notes                     taught you how)
──────────────────────────────────────────────────────────────────────
```

Two etiquette rules that cover 90% of cases:

1. **Ask questions that show your work.** "I reproduced Fig. 2 of X and got 3.1 instead
   of 2.8 — here's my repo, has anyone seen this gap?" gets answered. "How do I learn
   AI?" does not.
2. **Be generous before you need anything.** Answer beginners' questions (you are ahead
   of most people already), star and file good issues, credit people. The community has
   a long memory for both generosity and its absence.

---

## Open source: the beginner's credential

You cannot list a PhD you don't have. You CAN list commits, and everyone can verify them
in ten seconds. Open-source contribution is the most credential-blind path in the field:

- **Reproduce papers publicly.** You already did this in
  [lesson 15](../15-reproducing-a-paper/lesson.md). A repo titled "Reproducing [paper]:
  what matched, what didn't" with a clear README is *rare* and *valued* — most people
  never publish their reproductions. Three of these is a portfolio.
- **Contribute to nanoGPT-style repos.** Small, readable training repos (nanoGPT,
  llm.c-style projects, minimal implementations of new papers) welcome fixes: a bug, a
  clearer comment, a missing ablation, a speed improvement with a benchmark. Start with
  issues labeled "good first issue"; graduate to "I ran the ablation the README wondered
  about, here are the numbers."
- **Write READMEs like mini-papers.** What this is, the result (with a table), how to
  run it in one command, what's known-broken. Most repos fail this; yours won't, because
  lesson 28 taught you the structure. A clear README is the most visible writing sample
  you will ever produce.

Why this works: hiring managers and PhD advisors both face the same problem — predicting
whether you can do sustained, careful, honest technical work. A public trail of
commits, experiments, and writeups is *direct evidence*. A transcript is a proxy.

---

## Blogging: the distill path

A specific, repeatable pattern has launched many careers: **explain one thing so clearly
that experts share it.**

- distill.pub proved explainers count as research contributions.
- Jay Alammar's "The Illustrated Transformer" made him one of the most-cited educators
  in the field — it started as a blog post.
- Anthropic's interpretability team publishes research as web essays; several hires had
  blogs, not PhDs.
- Countless "I trained X on Y and here's what surprised me" posts have led directly to
  job conversations.

You have an unfair advantage here: you built everything from scratch across
[both](../../lessons/) [tracks](../README.md). You know exactly where beginners get
stuck, because you were recently stuck there. Experts can no longer see those spots.
That perspective is worth publishing.

---

## The career paths, mapped honestly

```
                        needs degree?   needs US/EU?   time to entry
PhD                     yes (bachelor)  no (global)    4–6 yrs
Industry research lab   usually no*     often no       1–3 yrs of public work
  (via residency/RE)
Independent/collective  no              no             immediately
Research engineer       no              often remote   1–2 yrs
```
\* for research-engineer and residency routes; research-scientist roles often expect a
PhD *or* an exceptional public record.

### PhD — what it actually is

Not "more school". It is a 4–6 year apprenticeship: you join a lab, and after year one
your life is research — reading, running experiments, writing papers, getting rejected,
resubmitting. The good: total immersion, a mentor, a cohort, and (in most programs) a
stipend — you are paid, modestly. The hard: the stipend is modest, progress is
nonlinear, and your experience depends enormously on one person (the advisor — choose
the advisor, not the university ranking).

**How admissions actually weigh things:** research experience and strong recommendation
letters from people who supervised research >> publications > grades > test scores >
prestige of undergrad. This is good news for you: a public record of real experiments
and a professor you've collaborated with online can beat a perfect GPA from anywhere.
Students from every continent get in this way; many programs (Europe, Canada, Asia)
also cost nothing and admit internationally as a matter of course. Cold-emailing a
professor with "I reproduced your paper, found X, here's the repo" is a known-successful
opening move.

### Industry research labs — residencies and fellowships

Labs run structured on-ramps precisely because they know talent doesn't all come through
PhDs:

- **Anthropic Fellows** program — safety-focused research sprints with mentorship.
- **Google / DeepMind** student-researcher and residency-style programs.
- **MATS** (ML Alignment Theory Scholars) — the standard on-ramp into safety research;
  explicitly takes people from non-traditional backgrounds; alumni are all over the
  safety labs.
- Various lab internships and "research collaborator" arrangements that started as a
  GitHub issue thread.

These are competitive, but the selection criterion is demonstrated research ability —
exactly the artifact trail this curriculum has you building. Application cycles recur;
treat a rejection as "not yet" and keep the public work compounding. Some require
relocation; a growing number are remote or have remote phases.

### Independent research and open collectives

**EleutherAI's story is the proof of concept.** In 2020 it was literally a Discord
server of people with no institutional backing who decided to replicate GPT-3-class
models in the open. They produced The Pile, GPT-Neo/GPT-NeoX, and dozens of papers.
Members went on to jobs at top labs — or stayed independent. Nobody asked for
credentials; contributions were the credential.

The pattern repeats: open collectives (EleutherAI, LAION, ML Collective, and successors)
run real research with volunteers, and they are the single most accessible entry point
on this list: join the Discord, pick up a task, do it well. Independent research is
also increasingly fundable (small grants for safety and open-source work exist and are
applied to with — again — a public track record).

### Research engineer vs research scientist

Do not be confused by titles:

- **Research scientists** typically set questions and lead papers. Often (not always)
  PhD-holders.
- **Research engineers** build the training runs, the infrastructure, the experiments —
  and at good labs they co-author papers, propose ideas, and drift into scientist roles.
  RE is *research*, and it is the **most accessible entry**: hiring is on demonstrated
  engineering + experimental skill, i.e., exactly a strong GitHub.

If you take one sentence from this section: **the RE path judges you on what this
curriculum taught you to produce.**

---

## How people actually get noticed

Not by credentials. By **consistent public work in one area**:

1. Pick a lane narrow enough to own (e.g., "tiny-model training dynamics",
   "attention-variant ablations", "interpretability of char-LMs").
2. Ship something public every few weeks: a reproduction, an ablation, an explainer, a
   negative result ([lesson 21](../21-negative-results/lesson.md) — publishing these is
   rare and memorable).
3. Show your work where the field looks: post the repo/plot on X, the writeup on your
   blog, the model on Hugging Face, the question in the right Discord.
4. Repeat for 6–18 months.

The compounding is real: work → tiny audience → feedback → better work → collaborators →
opportunities. Every step is visible to you, none requires permission, and geography
mostly doesn't matter — the community's shared language is English and its shared office
is the internet.

**Realistic timelines** (assuming ~10 focused hours/week of public work):

- 3 months: first reproduction repo + first explainer post. A few dozen readers. Normal.
- 6–12 months: a small body of work in one lane; first meaningful interactions with
  researchers; maybe a first contribution merged upstream.
- 12–24 months: known-in-your-niche; residency/MATS applications are credible; RE
  interviews reachable; PhD applications strong if that's your path.

Slower than the hype, faster than a degree.

---

## The exercise

See [exercise.md](exercise.md) — stop reading about presence and build it: your public
repo, your first post, an engagement plan, and a 6-month milestone plan.

## What's next
[Lesson 30 → Capstone: A Real Mini Research Project](../30-capstone-research-project/lesson.md)
