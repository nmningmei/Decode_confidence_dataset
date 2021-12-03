readme_Matthews_unpub.txt
Contributor: Julian Matthews (julian.r.matthews@gmail.com)

Citation: Matthews, J., Nagao, K. J., Ding, C., Newby, R., Kempster, P., & Hohwy, J. (2018, September 18). Impaired perceptual sensitivity with intact attention and metacognition in functional motor disorder. https://doi.org/10.31234/osf.io/fz3j2

Stimulus: A dual-task paradigm with central and/or peripheral stimulus detection. Stimulus_1 corresponds to the peripheral stimulus (a Gabor patch) and is coded for presence (code: 1) or absence (code: 0) of the target. Stimulus_2 corresponds to the central stimulus (an array of 4 rotated letters) and is coded for presence of a target letter ‘T’ (code: 1) or absence of this target (code: 0). Response_1 and Response_2 indicate the corresponding judgment or presence or absence by the participant for each stimulus type. Central and peripheral stimuli appeared on every trial (counterbalanced) but to avoid analysis errors I have coded the stimulus/response as ‘NaN’ if it was not task-relevant in the given block. 

Confidence scale: Participants made a simultaneous decision and confidence rating. Confidence was rated on a 4-point scale from 1 to 4. Participants were instructed that a rating of 1 corresponded to a ‘complete guess’ and a rating of 4 corresponded to ‘absolutely certain’. No explicit verbal instructions were given for confidence ratings of 2 or 3 but participants were encouraged to use the full confidence scale. Written cues for rating 1 (‘complete guess’) and 4 (‘absolutely certain’) were presented on the screen.

Manipulations: There were two within-subjects manipulations: Attention and Expectations. Attention was manipulated such that in the full attention condition (Condition_Attention code: ‘full_attention’) participants focussed on just the central (Task code: ‘central_task’) or peripheral (Task code: ‘peripheral_task’) stimulus. In the diverted attention condition (Condition_Attention code: ‘diverted_attention’) participants focussed on both stimuli at once (Task code: ‘dual_tasks’). Participants were instructed about the attention condition at the beginning of each block. Expectations were manipulated with respect to the presence of the peripheral stimulus. The peripheral target appeared either infrequently (on average 1 in every 4 trials; Condition_Expectations code: 0.25), half the time (Condition_Expectations code: 0.5), or frequently (Condition_Expectations code: 0.75). Participants were instructed about the expectations condition at the beginning of each block and were reminded after each trial.
There was one between-subjects manipulation: group. Participants were either healthy control participants (Group code: ‘control’) or belonged to one of two possible clinical cohorts. A team of neurologists classified clinical participants into those with a diagnosis of functional movement disorder (Group code: ‘functional_movement_disorder’) or organic movement disorder (Group code: ‘organic_movement_disorder’). 

Block size: Each block was 12 trials in length. Each participant completed 24 training blocks and 36 experimental blocks.
Feedback: Participants received no trial-by-trial feedback.

NaN fields: As explained above, I have coded Stimulus/Response/Confidence/etc cells ‘NaN’ if the corresponding stimulus was not task-relevant in a given block. No reaction time data was recorded for the central task so, for completion sake, this variable (RT_decConf_2) has been included in the data file but coded NaN.

Subject population: Age range 20-82 but see preprint for precise group demographics (https://psyarxiv.com/fz3j2/). As explained above, participants were either healthy control participants (Group code: ‘control’) or belonged to one of two possible clinical cohorts. A team of neurologists classified clinical participants into those with a diagnosis of functional movement disorder (Group code: ‘functional_movement_disorder’) or organic movement disorder (Group code: ‘organic_movement_disorder’).

Response device: Mouse.

Experimental setting: In individual room at hospital.

Special instructions: Subjects were encouraged to use the full confidence scale.

Link to material/codes: https://github.com/julian-matthews/FMD-public-repository

Experiment dates: 2017-2018.

Location of data collection: Monash Medical Centre, Clayton, AUSTRALIA.

Language of study: English.

Experimenter: Practicing neurologist trained in delivering experiment but otherwise new to psychophysics.