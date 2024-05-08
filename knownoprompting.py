openai_api_key = "your-key"

### SETUP ###
import openai
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# Load calibration and datset flag
load_set = False

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
#@title
# Load the 300 scenarios
scenario_info_path = 'fail.txt'
with open(scenario_info_path, 'r') as f:
  scenario_info_text = f.read()
scenario_info_text = scenario_info_text.split('\n\n')
print('Loaded scenario info from ' + scenario_info_path)

#@title
# Print a scenario
print('Sample scenario:\n')
print(scenario_info_text[0].split('\n',1)[1])  # remove the printed index

#@title
# Load the corresponding prompts for multiple choice generation
mc_gen_prompt_path = 'metabot-mc-gen-prompt.txt'
with open(mc_gen_prompt_path, 'r') as f:
  mc_gen_prompt_all = f.read().split('--0000--')
print('Loaded multiple choice generation prompts from ' + mc_gen_prompt_path)

#@title
# Print a prompt for MC generation
print(mc_gen_prompt_all[0])

#@title
# Split data into calibration set and test set
calibration_set = []
num_calibration_data = 200
num_test_data = 100
for i in range(num_calibration_data):
  calibration_set.append({
      'info': scenario_info_text[i],
      'mc_gen_prompt': mc_gen_prompt_all[i],
  })
test_set = []
for i in range(num_calibration_data, 300):
  test_set.append({
      'info': scenario_info_text[i],
      'mc_gen_prompt': mc_gen_prompt_all[i],
  })

  #@title
# Split data into calibration set and test set

for i in range(num_calibration_data):
  calibration_set[i]['info'] = scenario_info_text[i]

  
for i in range(num_calibration_data, 300):
  test_set[i-200]['info'] = scenario_info_text[i]


### MULTIPLE CHOICE QUESTION ANSWERING ###
#@markdown Print an example of task and prompt for MC generation
print('Sample task:\n')
print(calibration_set[0]['info'].split('\n',1)[1])  # remove the printed index
print('\n-------------\nPrompt for MC generation:\n')
print(calibration_set[0]['mc_gen_prompt'])


#@markdown Run MC generation (this may take a few minutes depending on API traffic)
import tqdm.notebook as tqdm

#!pip install ipywidgets widgetsnbextension
#!jupyter nbextension enable --py widgetsnbextension

import ipywidgets
from ipywidgets import FloatProgress
from ipywidgets import IntProgress
from IPython.display import display
#from ipywidgets import I
import pickle



print('Running calibration set...printing first five results...')
for i in range(num_calibration_data):
  calibration_data = calibration_set[i]
  prompt = calibration_data['mc_gen_prompt']
  _, text = lm(prompt, logit_bias={})
  text = text.strip()
  print(prompt)
  print(text)
  print('----')
  # print a few results
  if i < 5:
    #print(text)
    #print('----')
    print()
  calibration_data['mc_gen_raw'] = text
  calibration_set[i] = calibration_data

print('Running test set...printing first five results...')
for i in range(num_test_data):
  test_data = test_set[i]
  prompt = test_data['mc_gen_prompt']
  _, text = lm(prompt, logit_bias={})
  text = text.strip()
  test_data['mc_gen_raw'] = text
  test_set[i] = test_data


import pickle
num_calibration_data = 200
num_test_data = 100

# Load the list from the file
if load_set:
    with open('calibration_set4.pkl', 'rb') as f:
        calibration_set = pickle.load(f)

    with open('test_set4.pkl', 'rb') as f:
        test_set = pickle.load(f)


#@markdown Post-process the generated MC and get prompt for querying likelihood
import pickle
mc_score_background_prompt = "You are a robot operating in an office kitchen. "\
"You are in front of a counter with two closed drawers, a top one and a middle "\
"one. There is also a landfill bin, a recycling bin, and a compost bin."

def process_mc_raw(mc_raw, add_mc='an option not listed here'):
  mc_all = mc_raw.split('\n')

  mc_processed_all = []
  for mc in mc_all:
      mc = mc.strip()
      print(mc)
      #print(len(mc))

      # skip nonsense
      if len(mc) < 5 or mc[0] not in [
          'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
      ]:
          continue
      mc = mc[2:]  # remove a), b), ...
      mc = mc.strip().lower().split('.')[0]
      mc_processed_all.append(mc)

  print(mc_processed_all)
  print(len(mc_processed_all))
  #if len(mc_processed_all) < 4:
      #raise ValueError('Cannot extract four options from the raw output.')

  # Check if any repeated option - use do nothing as substitue
  mc_processed_all = list(set(mc_processed_all))
  #print(mc_processed_all)
  if len(mc_processed_all) < 4:
      num_need = 4 - len(mc_processed_all)
      for _ in range(num_need):
          mc_processed_all.append('do nothing')
  prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
  if add_mc is not None:
      mc_processed_all.append(add_mc)
      prefix_all.append('E) ')
  random.shuffle(mc_processed_all)

  # get full string
  mc_prompt = ''
  for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
      mc_prompt += prefix + mc
      if mc_ind < len(mc_processed_all) - 1:
          mc_prompt += '\n'
  add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
  return mc_prompt, mc_processed_all, add_mc_prefix

# process
print(calibration_set)
for dataset in [calibration_set, test_set]:
  for i in range(len(dataset)):
    #print(dataset[i]['mc_gen_raw'])
    mc_gen_raw = dataset[i]['mc_gen_raw'].strip()
    mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw)

    # get the part of the current scenario from the previous prompt
    cur_scenario_prompt = dataset[i]['mc_gen_prompt'].split('\n\n')[-1]
    limitation = dataset[i]['info'].split('Robot limitation: ')[1].split('\n')[0].lower()
    #print(limitation)

    # get new prompt
    mc_score_prompt = mc_score_background_prompt + '\n\n' + cur_scenario_prompt + '\n' + mc_gen_full
    mc_score_prompt += "\nWe: Which option is correct? Please answer with a single capital letter from either A, B, C, D, or E. Do not output anything else."
    mc_score_prompt += "\n You also have a physical limitation that prevents you from executing certain tasks. Your physical limitation: "
    mc_score_prompt += limitation
    mc_score_prompt += "\nYou:"
    dataset[i]['mc_score_prompt'] = mc_score_prompt
    #print(mc_score_prompt)

    # save other data
    dataset[i]['mc_gen_full'] = mc_gen_full
    dataset[i]['mc_gen_all'] = mc_gen_all
    dataset[i]['add_mc_prefix'] = add_mc_prefix


#@markdown Print an example of prompt for querying likelihood
print(calibration_set[0]['mc_score_prompt'])

#@markdown Query the likelihood of choice tokens (this may take a few minutes depending on API traffic)
num_tokens = 20
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
    limitations = info.split('Limitations: ')[1].split('\n')[0].lower()
    limitations = limitations.split(', ')
    limitations = [limitation.strip() for limitation in limitations]
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

    hw_options = [1] * 5

    # hardware clean
    for mc_ind, mc in enumerate(mc_gen_all):
        possible = True
        for limitation in limitations: 
            #print(limitation)
            if limitation in mc and len(limitation) != 0: 
              possible = False
              hw_options[mc_ind] = 0
        #if possible:
            #hw_options.append(token_all[mc_ind])
      

    # save
    #print(mc_gen_all)
    dataset[data_ind]['true_options'] = true_options
    #print(true_options)
    #print(hw_options)
    dataset[data_ind]['hw_options'] = hw_options

#@markdown Then, get the non-conformity scores from the calibration set, which is 1 minus the likelihood of the **true** option, $1-f(x)_{y_\text{true}}$. If there is more than one acceptable option for a scenario, we will use the minimum.
letters = ['A', 'B', 'C', 'D', 'E']
num_tokens = 20
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
  hw_options = data['hw_options']
  if count < 5: 
     print(top_logprobs)
     print(top_tokens)
     print(true_options)

  # getting logprobs of the mc from the top 20 generated by LLM

  top5_logprobs = np.ones(5) * (-50.0)


  for token_ind, token in enumerate(reversed(top_tokens)):
     if token in letter_to_index:
        top5_logprobs[letter_to_index[token]] = top_logprobs[num_tokens - 1 - token_ind]
  
  #print(top5_logprobs)

  count += 1
  # normalize the five scores to sum of 1
  mc_smx_all = temperature_scaling(top5_logprobs, temperature= 2)
  print(mc_smx_all)
  
  masked_smx_all = mc_smx_all * hw_options
  # get the softmax value of true option
  true_label_smx = [mc_smx_all[token_ind]
                    for token_ind, token in enumerate(letters)
                    if token in true_options]
  #print(true_label_smx)
  if(len(true_label_smx) != 0):
    true_label_smx = np.max(true_label_smx)
  else:
     true_label_smx = 0

  # get non-comformity score
  non_conformity_score.append(1 - true_label_smx)

print(non_conformity_score)



#@markdown Find q-hat as the quantile of the non-conformity scores, where N is number of calibration data. Also plot the histogram of the scores and the quantile.
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
hw_num_coverage = 0
num_help = 0
hw_num_help = 0
num_success = 0
hw_num_success = 0
set_size_all = []
for data in test_set:
  top_logprobs = data['top_logprobs']
  top_tokens = data['top_tokens']
  true_options = data['true_options']
  hw_options = data['hw_options']

  top5_logprobs = np.ones(5) * (-50.0)


  for token_ind, token in enumerate(reversed(top_tokens)):
     if token in letter_to_index:
        top5_logprobs[letter_to_index[token]] = top_logprobs[num_tokens - 1 - token_ind]

  # normalize the five scores to sum of 1
  mc_smx_all = temperature_scaling(top5_logprobs, temperature=2)
  prediction_options = mc_smx_all.copy()
  #print(prediction_options)
  #print(hw_options)
  masked_options = prediction_options * hw_options

  # include all options with score >= 1-qhat
  prediction_set = [
            token for token_ind, token in enumerate(letters)
            if mc_smx_all[token_ind] >= 1 - qhat
        ]
  execution_set = [
            token for token_ind, token in enumerate(letters)
            if masked_options[token_ind] >= 1 - qhat
        ]
  #print(prediction_set)
  #print(execution_set)
  set_size_all.append(len(prediction_set))

  # check coverage
  flag_coverage = not set(prediction_set).isdisjoint(true_options)
  num_coverage += flag_coverage

  hw_flag_coverage = not set(execution_set).isdisjoint(true_options)
  hw_num_coverage += hw_flag_coverage

  # check help - if prediction set is not singleton, or set include option E
  flag_help = len(prediction_set) != 1 or data['add_mc_prefix'] in prediction_set
  num_help += flag_help

  hw_flag_help = len(prediction_set) != 1 or data['add_mc_prefix'] in execution_set
  hw_num_help += hw_flag_help

  # check success - same as coverage
  num_success += flag_coverage
  hw_num_success += hw_flag_coverage

# get average rate
coverage_rate = num_coverage / num_test_data
help_rate = num_help / num_test_data
execution_help_rate = hw_num_help / num_test_data
success_rate = num_success / num_test_data
execution_success_rate = hw_num_success / num_test_data
print(hw_num_success)
print(execution_success_rate)
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
print('Execution help rate:', execution_help_rate)
print('Plan success rate:', success_rate)
print('Execution success rate:', execution_success_rate)
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