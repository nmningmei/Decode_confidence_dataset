Contributor: David Soto (d.soto@bcbl.eu)

Citation: Mei, N., Rankine, S., Olaffson, E., Soto, D. Predicting human prospective beliefs and decisions to engage using multivariate classification analyses of behavioural data. https://www.biorxiv.org/content/10.1101/607069v3

Category: Visual perception

Stimulus: On each trial, participants were required to discriminate the orientation of a brief, masked Gabor, presented at the threshold of visual awareness. Left tilted stimuli are coded as 1 and right tilted as 2. Left tilted responses are coded as 1 and right tilted responses as 2. In addition, prior to the discrimination response, participants were asked to report on the state of visual awareness of the stimulus. Following the discrimination response, they also rated the confidence in the correctness of the discrimination response.

Confidence scale:  

Both awareness and confidence reports were collected on each trial.

Regarding the awareness response, participants were asked to report 'unseen' (coded as 1) when they had no experience of the Gabor or saw a brief glimpse but were not aware of the orientation, and 'seen' when they could see the Gabor’s orientation somewhat or almost clearly (coded as 2). 

Regarding the confidence report, participants were instructed to report how confident they were about the correctness of the orientation response on a relative scale of 1 (relatively less) to 2 (relatively more) confident (code as 1 and 2). Participants were instructed that confidence ratings should be conceived in a relative fashion and hence they were asked to use all the confidence ratings independently of the awareness rating so that participants would not simply choose (e.g.) low confidence every time they were unaware and vice-versa on aware trials.

Manipulations: Prior to the presentation of the Gabor target, participants were instructed to report their  decision to engage attention on the upcoming trial.  Specifically, participants reported their decision to engage in a more or less focussed attentional state on the upcoming trial (coded as 2 or 1 respectively in the variable 'Attention_Decision').

Feedback: No

NaN fields: There should be no NaNs when loading the data. There were no response deadlines and all answers were recorded.

Block size: There were 12 blocks of 50 trials each. 

Trial count: Please note this starts with 0, so 0 is trial 1.

Subject population: Participants were healthy volunteers aged between 18 and 47 years (mean age 21.6; 9 males). Standard exclusion criteria were applied as indicated in the paper. However, all the original raw data files could not be collated, likely due to the senior author moving research institutions. Only the subjects included in the analyses appear in the .csv files.

Response device: Keyboard.

Experiment setting: Laboratory.

Training: In each of the Experiments, there was a short practice with visible stimuli to familiarise participants with the discrimination task. A calibration phase of the luminance of the Gabor followed. Then, participants performed 15 training trials identical to the experimental trials.

Experiment goal: The goals of the study were (i) to replicate in a different context the observations made in Jachs et al. (2015, JEP: HPP) that type-1 perceptual sensitivity is more strongly related to visual awareness than type-2 metacognitive sensitivity; (ii) to assess whether the different prospective decisions play a functional role in shaping subsequent perception or metacognitive performance, namely, type-1 and type-2 sensitivity; (iii) most critically, we wanted to understand the factors that determine the prospective decisions to engage with the task at hand in a particular way (i.e. to adopt a more or less focussed attentional state). Standard machine learning classifiers were used to predict the decision to engage attention on the current trial given a vector of features (i.e. task correctness, visual awareness, and confidence) from the previous trials, considering 1-back, 2-back, 3-back, and 4-back trials separately.

Main result: We found that the current prospective decision to engage attention could be predicted from information concerning task-correctness, stimulus visibility and response confidence from previous trials. Like in Experiment 1, awareness and confidence were more diagnostic of the prospective decision than task correctness. Notably, classifiers trained with prospective beliefs of success (Experiment 1) predicted decisions to engage attention and vice-versa. We also replicated the observations made by Jachs et al. (2015). We also found that prospective decisions to engage attention were followed by better self-evaluation of the correctness of discrimination responses.  

Special instructions: Participants were instructed that confidence ratings should be conceived in a relative fashion and hence they were asked to use the confidence ratings independently of the awareness rating so that participants would not simply choose low confidence every time they were unaware and vice-versa on aware trials. They were also instructed to respond as precisely as possible in all the responses. There was no response deadline and speed was not emphasized.

Experiment dates: Data was collected in April 2015.