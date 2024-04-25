openai_api_key = "your-api-key"

import openai
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import pickle

### SETUP ###
# Set OpenAI API key.
openai.api_key = openai_api_key
client = OpenAI(api_key = openai_api_key)

#@markdown LLM API call
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def lm(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       stop_seq=None,
       logit_bias={
          317: 100.0,   #  A (with space at front)
          347: 100.0,   #  B (with space at front)
          327: 100.0,   #  C (with space at front)
          360: 100.0,   #  D (with space at front)
          412: 100.0,   #  E (with space at front)
          628: -100.0,
          198: -100.0,
      },
       timeout_seconds=20):
  max_attempts = 5
  for _ in range(max_attempts):
      try:
          with timeout(seconds=timeout_seconds):
              response = client.completions.create(
                  model='gpt-3.5-turbo-instruct',
                  prompt=prompt,
                  max_tokens=max_tokens,
                  temperature=temperature,
                  logprobs=logprobs,
                  logit_bias=logit_bias,
                  stop=list(stop_seq) if stop_seq is not None else None,
              )
          break
      except:
          print('Timeout, retrying...')
          pass
  return response, response.choices[0].text.strip()


### LOAD PRE-GENERATED DATASET ###

# Load the list from the file
with open('calibration_set.pkl', 'rb') as f:
    calibration_set = pickle.load(f)

with open('test_set.pkl', 'rb') as f:
    test_set = pickle.load(f)

num_calibration_data = 200
num_test_data = 100



#@markdown Print an example of prompt for querying likelihood
#print(calibration_set[0]['mc_score_prompt'])


#@markdown Query the likelihood of choice tokens (this may take a few minutes depending on API traffic)
num_tokens = 10
temp = 1
for i, dataset in enumerate([calibration_set, test_set]):
  if i == 0:
    print('Running calibration set...printing first five results...')
  else:
    print('Running test set...printing first five results...')
  for i in range(len(dataset)):
    prompt = dataset[i]['mc_score_prompt']

    # call LLM API
    mc_score_response, _ = lm(prompt, max_tokens=1, logprobs=num_tokens)
    #print(mc_score_response.choices[0].text)
    top_logprobs_full = mc_score_response.choices[0].logprobs.top_logprobs[0]
    top_tokens = [token.strip() for token in top_logprobs_full.keys()]
    top_logprobs = [value for value in top_logprobs_full.values()]
    if i < 5:
      print(top_tokens, top_logprobs)
    dataset[i]['top_logprobs_full'] = top_logprobs_full
    dataset[i]['top_tokens'] = top_tokens
    dataset[i]['top_logprobs'] = top_logprobs



### SPECIFY TARGET SUCCESS LEVEL ###
target_success = 0.8  #@param {type: "number"} {form-width: "10%"}
epsilon = 1-target_success
print('Epsilon:', epsilon)



### CALIBRATION WITH CONFORMAL PREDICTION ###
#@markdown First, for convenience, we use simple heuristics to determine if each option generated is a true option. In the pre-generated data, we defined the user intent (object to be moved, and the target location). In the paper for the Mobile Manipulation example, the true options are labelled manually.
for dataset in [calibration_set, test_set]:
  for data_ind, data in enumerate(dataset):
    true_options = []
    info = data['info']
    mc_gen_all = data['mc_gen_all']

    # extract the line started with user intent (object) from info
    info = info.split('\n',1)[1]  # remove index
    true_obj = info.split('User intent (object): ')[1].split('\n')[0].lower()
    true_obj = true_obj.split(',')
    true_obj = [obj.strip() for obj in true_obj]
    true_target_loc = info.split('User intent (location): ')[1].split('\n')[0].lower()
    # print(true_obj, '   ', true_target_loc)

    # go through all mc
    token_all = ['A', 'B', 'C', 'D', 'E']
    for mc_ind, mc in enumerate(mc_gen_all):
      if 'not listed here' in mc or 'do nothing' in mc: continue

      # corner case: if there is only one sponge, shrink true_obj to only sponge
      scene_obj = info.split('Scene objects:')[1].split('\n')[0].split(', ')
      scene_obj = [obj.strip().lower() for obj in scene_obj]
      for i in range(len(true_obj)):
        if not ('clean sponge' in scene_obj and 'dirty sponge with food residue' in scene_obj) \
          and 'sponge' in true_obj[i]:
          true_obj[i] = 'sponge'

      # corner case: mc has both clean and dirty
      if 'clean' in mc and 'dirty' in mc: continue

      # corner case: orange and orange soda - not dealt with

      if true_target_loc == 'pick-up':
        # check if more than one scene object in the mc
        num_obj_in_mc = 0
        for obj in scene_obj:
          if obj in mc.lower(): num_obj_in_mc += 1
        if num_obj_in_mc > 1: continue

        for obj in true_obj:
          if obj in mc and 'drawer' not in mc and 'bin' not in mc and \
            'microwave' not in mc and 'cooktop' not in mc:
            true_options.append(token_all[mc_ind])
      elif 'drawer' in true_target_loc:
        for obj in true_obj:
          if obj in mc and true_target_loc in mc:
            true_options.append(token_all[mc_ind])
      elif 'recycling' in true_target_loc or 'landfill' in true_target_loc \
        or 'compost' in true_target_loc or 'microwave' in true_target_loc \
        or 'cooktop' in true_target_loc:
        for obj in true_obj:
          if obj in mc and true_target_loc in mc:
            true_options.append(token_all[mc_ind])
      else: # target location is an object
        mc_obj_pick_up_phrase = mc.split('and')[0]
        mc_obj_place_phrase = mc.split('and')[1]
        for obj in true_obj:
          if obj in mc_obj_pick_up_phrase and true_target_loc in mc_obj_place_phrase:
            true_options.append(token_all[mc_ind])

    # if none correct
    if len(true_options) == 0:
      true_options = [data['add_mc_prefix']]

    # save
    dataset[data_ind]['true_options'] = true_options


#@markdown Then, get the non-conformity scores from the calibration set, which is 1 minus the likelihood of the **true** option, $1-f(x)_{y_\text{true}}$. If there is more than one acceptable option for a scenario, we will use the minimum.
letters = ['A', 'B', 'C', 'D', 'E']
letter_to_index = {letter: index for index, letter in enumerate(letters)}

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

non_conformity_score = []
token_all = ['A', 'B', 'C', 'D', 'E']
count = 0
for data in calibration_set:
  top_logprobs = data['top_logprobs']
  #print(top_logprobs)
  top_tokens = data['top_tokens']
  #print(top_tokens)
  true_options = data['true_options']
  #print(true_options)


  # getting logprobs of the mc from the top 20 generated by LLM
  top5_logprobs = np.ones(num_tokens) * (-15.0)

  for token_ind, token in enumerate(reversed(top_tokens)):
     if token in letter_to_index:
        top5_logprobs[letter_to_index[token]] = top_logprobs[num_tokens - 1 - token_ind]
  
  if count < 5: 
     print(top_logprobs)
     print(top_tokens)
     print(true_options)
     print(top5_logprobs)

  count += 1
  # normalize the five scores to sum of 1
  mc_smx_all = temperature_scaling(top5_logprobs, temperature= temp)
  print(mc_smx_all)
  
  # get the softmax value of true option
  true_label_smx = [mc_smx_all[token_ind]
                    for token_ind, token in enumerate(letters)
                    if token in true_options]
  print(true_label_smx)
  if(len(true_label_smx) != 0):
    true_label_smx = np.max(true_label_smx)
  else:
     true_label_smx = 0

  # get non-comformity score
  non_conformity_score.append(1 - true_label_smx)

print(non_conformity_score)


#@markdown Find $\widehat{q}$ as the $\frac{\lceil (N+1)(1-\epsilon) \rceil}{N}$ quantile of the non-conformity scores, where $N$ is number of calibration data. Also plot the histogram of the scores and the quantile.
q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
qhat = np.quantile(non_conformity_score, q_level, method='higher')
print('Quantile value qhat:', qhat)
print('')

# plot histogram and quantile
plt.figure(figsize=(6, 3))
plt.hist(non_conformity_score, bins=30, edgecolor='k', linewidth=1)
plt.axvline(
    x=qhat, linestyle='--', color='r', label='Quantile value'
)
plt.title(
    'Histogram of non-comformity scores in the calibration set'
)
plt.xlabel('Non-comformity score')
plt.legend(); plt.show()
print('')
print('A good predictor should have low non-comformity scores, concentrated at the left side of the figure')




### PREDICTION AT TEST TIME AND TRIGGERING HUMAN HELP ###
#@title
num_coverage = 0
num_help = 0
num_success = 0
set_size_all = []
for data in test_set:
  top_logprobs = data['top_logprobs']
  top_tokens = data['top_tokens']
  true_options = data['true_options']

  top5_logprobs = np.ones(num_tokens) * (-15.0)


  for token_ind, token in enumerate(reversed(top_tokens)):
     if token in letter_to_index:
        top5_logprobs[letter_to_index[token]] = top_logprobs[num_tokens - 1 - token_ind]

  # normalize the five scores to sum of 1
  mc_smx_all = temperature_scaling(top5_logprobs, temperature=temp)

  # include all options with score >= 1-qhat
  prediction_set = [
            token for token_ind, token in enumerate(letters)
            if mc_smx_all[token_ind] >= 1 - qhat
        ]
  print(prediction_set)
  set_size_all.append(len(prediction_set))

  # check coverage
  flag_coverage = not set(prediction_set).isdisjoint(true_options)
  num_coverage += flag_coverage

  # check help - if prediction set is not singleton, or set include option E
  flag_help = len(prediction_set) != 1 or data['add_mc_prefix'] in prediction_set
  num_help += flag_help

  # check success - same as coverage
  num_success += flag_coverage

# get average rate
coverage_rate = num_coverage / num_test_data
help_rate = num_help / num_test_data
success_rate = num_success / num_test_data
avg_prediction_set_size = np.mean(set_size_all)

# show results
print('============== Summary ==============')
print('Number of calibration data:', num_calibration_data)
print('Number of test data:', num_test_data)
# print('Quantile value:', qhat)
print('Average prediction set size:', avg_prediction_set_size)
print('Marginal coverage guarantee:', 1 - epsilon)
print('Empirical coverage:', coverage_rate)
print('Help rate:', help_rate)
print('Success rate:', success_rate)
print('\nSuccess rate should be the same as empirical coverage, and it is close to the marginal coverage guarantee, 1-epsilon\n')

# plot histogram of prediction set size
plt.figure(figsize=(6, 3))
plt.hist(
    set_size_all, bins=np.arange(-1, 6) + 0.5, edgecolor='k',
    linewidth=1
)
ax = plt.gca()
ax.locator_params(integer=True)
plt.title('Histogram of prediction set size')
plt.xlabel('Prediction set size')
plt.ylabel('Frequency')
plt.show()



