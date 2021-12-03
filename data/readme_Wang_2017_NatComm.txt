Contributor: Shuo Wang (shuo.wang@mail.wvu.edu) and Sai Sun

Citation:
Wang S, Yu R, Tyszka JM, Zhen S, Kovach C, et al. 2017. The human amygdala parametrically encodes the intensity of specific facial emotions and their categorical ambiguity. Nature Communications 8: 14821

Description of variables:
Stimulus Level: 1: 100% fearful, 2: 70% fearful, 3: 60% fearful, 4: 50% fearful, 5: 40% fearful, 6: 30% fearful, 7: 0% fearful.

Response: 1: judging face as fear, 2: judging face as happy, NaN: missing button press.

RT: Note that this task is not a speeded task, i.e., participants were instructed to press the button after stimulus offset (stimulus duration = 1s).

Confidence scale: 3: very sure, 2: sure, 1: unsure

Condition: participant type.

Participants:
Data from epilepsy neurosurgical patients and 3 amygdala lesion patients are included. There are 13 sessions from 8 neurosurgical patients in total (3 patients did two sessions and 1 patient did three sessions. Each session is treated as an independent sample for behavioral analysis. AP, AM and BG are three patients with selective bilateral amygdala lesions as a result of Urbach–Wiethe disease. Note that in the original publication there are 9 neurosurgical patients and 19 fMRI participants. Because they do not contribute to confidence ratings, we do not include their data here.

Detailed description of task and stimuli:
We asked participants to discriminate between two emotions, fear and happiness. We selected faces of four individuals (2 female) each posing fear and happiness expressions from the STOIC database, which are expressing highly recognizable emotions. Selected faces served as anchors, and were unambiguous exemplars of fearful and happy emotions as evaluated with normative rating data provided by the creators. To generate the morphed expression continua for this experiment, we interpolated pixel value and location between fearful exemplar faces and happy exemplar faces using a piece-wise cubic-spline transformation over a Delaunay tessellation of manually selected control points. We created 5 levels of fear-happy morphs, ranging from 30% fear/70% happy to 70% fear/30% happy in a step of 10%. Low-level image properties were equalized by the SHINE toolbox (The toolbox features functions for specifying the (rotational average of the) Fourier amplitude spectra, for normalizing and scaling mean luminance and contrast, and for exact histogram specification optimized for perceptual visual quality). 

In a subset of epilepsy patients, in each block, each level of the morphed faces was presented 16 times (4 repetitions for each identity) and each anchor face was presented 4 times (1 for each identity). In all other epilepsy patients and amygdala lesion patients, each anchor face and each morphed face was presented the same number of times (12 times, 3 repetitions for each identity). Epilepsy patients performed 2 to 5 blocks, and amygdala lesion patients performed 4 blocks.

A face was presented for 1 second followed by a question prompt asking participants to make the best guess of the facial emotion, either by pushing the left button (using left hand) to indicate that the face was fearful, or by pushing the right button (using right hand) to indicate that the face was happy. Participants had 2 seconds to respond, otherwise the trial was aborted and there was a beep to indicate time-out. Participants were instructed to respond as quickly as possible after stimulus offset. After emotion judgment and a 500 ms blank screen, participants were asked to indicate their confidence of judgment. This question also had 2 seconds to respond. No feedback message was displayed either questions. An inter-trial-interval (ITI) was jittered between 1 to 2 seconds. The order of faces was completely randomized for each participant. Participants practiced 5 trials before the experiment to familiarize themselves with the task. In the end, the overall percentage of “correct answers” (calculated by the morph levels) was displayed to participants as a motivation.