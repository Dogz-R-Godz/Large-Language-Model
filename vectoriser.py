import neural_network as nn
import random as rand
import pickle
import gzip
import csv



three_letter_combos=[]
alphabet=[' ',"a","b","c","d","e","f","g",'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',',','.','/',';','<','>','?',':']
for letter in alphabet:
    for letter2 in alphabet:
        for letter3 in alphabet:
            three_letter_combos.append(f"{letter}{letter2}{letter3}")
#print(three_letter_combos)
middle=[]
for mid in range(100):
    middle.append(f"m{mid}")
network=nn.network(len(three_letter_combos),len(three_letter_combos),[middle])
make_new_network=False
if make_new_network:
    Network=network.randomise_network([-1,1],True)
    with open('vectorising_brain.pickle', 'wb') as handle:
        pickle.dump(Network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Successfully saved the brain!")
else:
    with open('vectorising_brain.pickle', 'rb') as handle:
        Network = pickle.load(handle)
    print("Successfully loaded the brain!")

all_sentences=[]

with gzip.open('GenericsKB-Best.tsv.gz', 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        all_sentences.append(row[3].lower())

update_counter=0
surrounding_tokens=[]
for sentence in range(len(all_sentences)):
    surrounding_tokens.append([])
    token=''
    counter=0
    for letter in all_sentences[sentence]:
        counter+=1
        token+=letter
        if counter==3:
            counter=0
            surrounding_tokens[sentence].append(token)
            token=''
    if counter>0:
        while len(token)<3:
            token+=' '
        surrounding_tokens[sentence].append(token)
    x=0
    update_counter+=1
    if update_counter>10000:
        print(f"{(sentence/len(all_sentences))*100}% Done!")
        update_counter=1
counter=0
tokens_to_train_on={} #go through each token, and find 10 tokens in each direction to it. Save them, and use them to train it.
for sentence in surrounding_tokens:
    counter+=1
    prevous_tokens=[]
    for token in range(len(sentence)):
        if sentence[token] in tokens_to_train_on:
            tokens_to_train_on[sentence[token]]+=prevous_tokens
            for new_token in range(min(len(sentence)-token,10)):
                tokens_to_train_on[sentence[token]].append(sentence[new_token+token])
        else:
            tokens_to_train_on[sentence[token]]=prevous_tokens
            for new_token in range(min(len(sentence)-token,10)):
                tokens_to_train_on[sentence[token]].append(sentence[new_token+token])
        if len(prevous_tokens)>=10:
            prevous_tokens.pop(0)
        prevous_tokens.append(sentence[token])
    if counter%20000==0:
        print(counter)

print("Done tokenising. Getting ready to neural network this thing")
counter=0
expected_outputs=[]
states=[]
base_state={}
base_expected_output={}
eos=[]
sta=[]
for state in range(len(three_letter_combos)):
    base_state[f"i{state}"]=0
    base_expected_output[f"o{state}"]=0
new_tokens=list(tokens_to_train_on.keys())
while len(new_tokens)>0 and counter<500:
    curr_state={}
    for state in range(len(three_letter_combos)):
        curr_state[f"i{state}"]=0
    counter+=1
    curr_token=rand.choice(new_tokens)
    
    for _ in range(3):
        curr_state=base_state.copy()
        curr_expected_output=base_expected_output.copy()
        
        
        tokens_new = list(dict.fromkeys(tokens_to_train_on[curr_token]))
        curr_token2=rand.choice(tokens_new)
        curr_state[f"i{tokens_new.index(curr_token2)}"]
            
        curr_expected_output[f"o{new_tokens.index(curr_token)}"]=0
        states.append(curr_state)
        expected_outputs.append(curr_expected_output)
    if counter%10==0 and counter!=0:
        eos.append(expected_outputs)
        sta.append(states)
        states=[]
        expected_outputs=[]
    new_tokens.pop(new_tokens.index(curr_token))
    
    if len(new_tokens)%20==0:
        print(f"Yoo {len(new_tokens)} left")
        

print("Finished getting the states and expected outputs. Running the backpropergation algorithm now.")
print(len(sta))
for state in range(len(sta)):
    Network=network.backpropergation(Network,eos[state],sta[state],1,0.1)