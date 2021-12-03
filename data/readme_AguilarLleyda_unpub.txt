Readme file for “data_AguilarLleyda_inprep.csv”

MAIN INFORMATION:
Contributors: David Aguilar-Lleyda, Mahiko Konishi, Jérôme Sackur, Vincent De Gardelle

Contact e-mail: aguilarlleyda@gmail.com

Citation: these data are not published in any form yet.

Experiment in paper: none yet, planned to be Experiment 1.

Stimulus: an array of 80 letters (upper-case O or X, colored blue or orange) were shown for one second. They were randomly disposed within an imaginary square, in a very similar fashion as Rahnev et al. (2015, Psych Science).
Perceptual choice and confidence rating mechanism: after the stimuli disappeared, participants had to report the dominant color and the dominant letter of the array they had just seen, as well as rating their confidence on each choice. First two horizontally-arranged boxes appeared. On some trials, a square contained a blue square and the other an orange square. Then participants chose the dominant color by means of the keyboard (E to choose the color appearing on the left box, R to choose that appearing on the right box, with the horizontal placement being randomized across trials). Then a vertical 50-to-100 (50 = total uncertainty / responding at chance, 100 = total certainty) confidence scale appeared and participants used their mouse to rate their confidence in the choice they had just made. Immediately after, and following the same procedure, participants reported the dominant letter and rated their confidence for that choice. On some other trials, the choice + confidence for the letter task was done first, and for color second.

Manipulations: on each trial, the color and the letter task's difficulty was manipulated independently. Each task could be easy (expected performance of 90% correct) or hard (expected performance of 60% correct). Within the color task, in some trials there were more blue than orange elements, and in others vice versa. The same for the letter task with Os and Xs.

Block size: Participants completed 4 runs of 4 blocks each, with 24 trials per block.

Feedback: after each block, participants were told the percentage of correct choices, pooled across tasks (without differentiating color and letter choices).


DATASET VARIABLES:
Subj_idx: subject identifier
Run: run number
Block: block number (within that run)
Trial: trial number (within that block)
Difficulty_col: expected performance for the color task, in % (90 = easy, 60 = hard)
Difficulty_let: expected performance for the letter task, in % (90 = easy, 60 = hard)
Number_blues: for the color task, number of blue elements, over a total of 80
Number_os: for the letter task, number of Os, over a total of 80
Stimulus_col: dominant dimension for the color task (1 = more blue elements were presented, 2 = more orange elements were presented)
Stimulus_let: dominant dimension for the letter task (1 = more Os were presented, 2 = more Xs were presented)
Response_col: response given for the color task (1 = blue, 2 = orange)
Response_let: response given for the letter task (1 = O, 2 = X)
Accuracy_col: accuracy for the color task (0 = error, 1 = correct response)
Accuracy_let: accuracy for the letter task (0 = error, 1 = correct response)
RT_dec_col: RT for the color task's perceptual choice
RT_dec_let: RT for the letter task's perceptual choice
Confidence_col: confidence rating for the color task (from 50 to 100)
Confidence_let: confidence rating for the letter task (from 50 to 100)
RT_confidence_col: RT for the confidence rating of the color task
RT_confidence_let: RT for the confidence rating of the letter task
Taskorder: order in which the responses + confidence had to be given (1 = color-then-letter, 2 = letter-then-color)
Locationchoice_col: horizontal position of the box that contained the dimension of the color task that the participant chose as dominant when making the perceptual choice (1 = left, 2 = right)
Locationchoice_let: horizontal position of the box that contained the dimension of the letter task that the participant chose as dominant when making the perceptual choice (1 = left, 2 = right)
Initposconf_col: initial random vertical position at which the cursor started on the confidence scale, when rating confidence for the color choice
Initposconf_let: initial random vertical position at which the cursor started on the confidence scale, when rating confidence for the letter choice

OTHER INFORMATION:
Subject population: consenting adults, age range approximately 18-50, that participated in exchange for an economic remuneration.

Response device: keyboard (perceptual choices) and mouse (confidence responses).

Experiment setting: lab. The room fitted up to 20 people at the same time.

Training: 5 trials where only the color choice (+ confidence) had to be made, 5 trials where only the letter choice (+ confidence) had to be made (these were counterbalanced). Then, participants completed 96 trials of doing both tasks (color and letter) at the same time, where an independent staircase procedure for each task was active.

Experiment goal: this experiment is intended to be a baseline for an upcoming project assessing confidence leak.

Main result: within-trial confidence leak was found.

Special instructions: responses were not speeded. Participants were told that they would be rewarded both on the basis of their choices and their confidence ratings.

Experiment dates: participants were recruited in Spring 2019.

Location of data collection: Laboratoire d'Économie Experimentale, Université Paris 1 Panthéon-Sorbonne, Paris, France. The experiment was conducted in French.

EXCLUSION CRITERIA
The design of the experiment that originated this dataset relied on a staircase procedure to obtain, for both tasks, easy (90% expected performance) and hard (60% expected performance) trials. It could be assumed that the staircase worked and that hard trials were indeed harder than easy trials, as reflected by observed performance. However, for whatever reason, the staircase seems to not have worked that well for some participants. That is why we recommend to filter the present (unfiltered) dataset by implementing some arbitrary exclusion criteria. We suggest to exclude a participant if any of the following conditions holds:Color easy does not have a performance at least 15% superior than color hard.Letter easy does not have a performance at least 15% superior than letter hard.Letter easy has a performance more than 15% or less than 15% that of color easy.Letter hard has a performance more than 15% or less than 15% that of color hard.
If the previous criteria are implemented, 4 participants should be removed.
    