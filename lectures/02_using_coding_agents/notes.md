Structure of this lecture:

 - introduction to coding agents
 - claude code specific skills:
    - plans and plan mode
    - documentation in claude md files
    - subagents
    - goals
    - worktrees
  - knowing what task to hand off
  - how to check wether the task was done correctly
  - how to work on things in a parallelizable way
  - how to have re-usable infrastructure 

 - have a phase where we set up our coding enveieernment and do a series of exercises desinged to teach the treatment of coding agents. These are branches of the researchscaffold repo:

  - change somem setup from one service into another
  - there is an example script, that does some process. make it config-controlled, so that all of the different parameters can be configered. then do a wandb sweep over them
  - some prompt based thing, like a scenatio, where the model could answer deceptively or something is in a database. the task is to come up with a bunch more of these scenarios, and to let claude work them out


Exercise for 90 min: replicate a paper:

we have the choice between two papers:
for Team A this morning: 
Replicate the first figure of this paper: https://www-cdn.anthropic.com/b9ca6db27f02a9ddf0d4fdb51b26432c99a27be0.pdf

with the reasoning models you have available on openrouter, reporduce the central plot that shows how faithful they are, measured on wether a flipped answer is mentiong the hint in the CoT.
This behaviour is brittle, especially with modern models. Try out different models, datasets and ways of phrasing the hint, to make it better understandable for you.
if you are finished early, try to expand your replication to also show the central result of this paper http://arxiv.org/abs/2507.05246, that more complex CoTs do not have this even.


for those in group B:
reproduce the central result of this paper: http://arxiv.org/abs/2406.11717
show that you can extract a single direction for refusals, and by ablating it you can zero out refusals, and when you add it, you can induce refusals.

once you have shown this, see how robust this effect is, what if the dataset of of prompts to generate the vector, and the prompts to test its refusal effect are more different.

