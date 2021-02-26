import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import time
from IPython.display import clear_output
from datetime import datetime
import re

import nltk
from nltk import FreqDist
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Uncomment the code below to download neccesary packages
# nltk.download("book")
# from nltk.book import *


def format_df(excel_path, excel_sheet):
    '''
    Used for start()
    Create a format suitable for analysis. 
    Input is the path to the excel file and the sheet name within the excel file
    '''
    df = pd.read_excel(excel_path, sheet_name=excel_sheet, engine='openpyxl')
    df=df.iloc[:,0:-6] #exclude last 6 columns
    num_questions=len(df.columns)-1 #get number of questions
    column_order=df.columns[1:num_questions+1]# used to generate question_number
    df=df.melt(id_vars=['StudyID']) #create long format
    df.columns=['StudyID','Question','Response'] #rename columns
    column_order=pd.DataFrame({'order':column_order}) #create new df to generate question_number
    column_order= column_order.reset_index().set_index('order')
    df['Question_Number']=df['Question'].map(column_order['index']) #map question numbers
    df['Question_Number']=df['Question_Number']+1 #change from 0-index to 1-index for question_number
    df=df.sort_values(by=['StudyID','Question_Number']) #sort df
    df=df.set_index(['StudyID']) #set StudyID as index
    df=df.loc[:,['Question','Question_Number','Response']]#reorder columns
    return df

def prep_word(word):
    '''
    helper function for top_words()
    used to lemmatize the output
    '''
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower())

def remove_stpwrd(wlist, s_words):
    '''
    Helper function for top_words()
    used to create stop words in top_words
    '''
    cust_stopwords = ['.', ',', 'i', 'it', 'my', 'to', 'me', 'the', 'into', 'ha', 'have', 'in', 'of', 'are', 'it', 'for', 'nan','the', 'is', 'its', 'and', 'a', "'s", "'m", 'been', 'at']# nan is probably for emojis
    wlist = [word for word in wlist if word not in cust_stopwords] # drop words in pre-defined stopwords
    wlist = [word for word in wlist if word not in s_words]# drop words in assigned stopwords
    return wlist


def top_words(cluster, num_words = 1, s_words=[]):
    '''
    Used in examine_topics() and start()
    this function yield top tokens for each cluster
    '''
    # print number of lines:
    print("Number of lines: {}".format(len(cluster)))
    # combine all lines:
    corpus = " ".join(cluster)
    # tokenize:
    corpus = nltk.word_tokenize(corpus)
    # preprocess the words: lemmatize, stem, get rid of stopwords:
    words = [prep_word(word) for word in corpus]
    # remove stopwords:
    words = remove_stpwrd(words, s_words)
    # output a vocab of tokens and their respective counts
    vocab = FreqDist(words)
    # set number to report
    n = min(len(vocab), num_words)
    # print top words
    print("Top {} words in cluster are {}" \
          .format(n, vocab.most_common(n)))
    
def cluster(df, encoded, cat = '0', n_clusters = 2):
    '''
    Used create_initial_cluster() and start()
    this function goes to category cat = 0 (or other values)
    apply clustering/ splitting
    replace the category value by its (3) sub-categories
    '''

    #get index of rows withcategory
    indices = df[df["category"] == cat].index.tolist()

    # use agglomerative clustering
    clustering = AgglomerativeClustering(distance_threshold = None, n_clusters = n_clusters, 
                                         linkage = "complete", affinity = 'cosine').fit(encoded[indices,:])
    # sub-categories
    df.loc[indices,"category"] = df.loc[indices,"category"] + "." + (clustering.labels_+1).astype(str)

    return df


def examine_topics(branches, rand_lines = 10, top_n_words = 20, s_words=[], show_top=True):
    '''
    Used in examine_branch() and start()
    print top words and random lines of branches:
    '''
    #loop through branches
    for topic in sorted(list(set(branches['category'])), reverse = False):
        # only print top words if it is requested
        if show_top:
            print("Topic {}".format(topic))
            top_words(branches[branches['category']==topic]["text"], num_words = top_n_words, s_words=s_words)
        print(' ')
        #calculate number lines to print and prints them
        n_lines = min(rand_lines, branches[branches['category']==topic].shape[0])
        count=n_lines #use to determine how many lines actually printed
        for line in branches[branches['category']==topic].sample(n_lines).text:
            if line != 'nan':
                print(line)
            else:
                count=count-1
        #let users know if no lines are printed
        if count ==0:
            print("No valid samples in this cluster")

def create_initial_clusters(df,q_num):
    '''
    Used start()
    Create the initial two clusters
    '''
    #creates model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    #select the responses
    df=df.loc[df.Question_Number == q_num, "Response"]

    #change bad responses to np.nan
    for i in range(len(df)):
        string=str(df.iloc[i])
        if (string.strip() =='' or string.lower()=='none' or string.lower()=='next' or string.lower()=='nan'):
            df.iloc[i]=np.nan

    # Select responses
    X_texts = df.astype(str).to_list()

    # encode only the specific list of texts:
    encoded = np.array(model.encode(X_texts))
    full_df = pd.DataFrame({'text': X_texts})

    #string of question number
    q_str=str(q_num)

    # set/reset all categories = question number
    full_df["category"] = q_str

    # at the beginning, only one category = question number
    full_df = cluster(full_df, encoded = encoded, n_clusters = 2, cat = q_str)

    #split question number into two more clusters
    full_df = cluster(full_df, encoded = encoded, n_clusters = 2, cat = q_str+'.1')
    full_df = cluster(full_df, encoded = encoded, n_clusters = 2, cat = q_str+'.2')
    
    return (full_df,encoded)

def get_groups(branches):
    '''
    Used in start()
    get all groups
    Similar to examine groups but has a return instead of print
    '''
    groups={}
    for topic in sorted(list(set(branches['category'])), reverse = False):
        topics=[]
        for line in branches[branches['category']==topic].text:   
            topics.append(line)
        groups[topic]=topics
    return groups #dictionary of lists of strings

def examine_branch(groups,branch,s_words=[], show_top=True):
    '''
    Used in start()
    examines/prints one branch of cluster
    '''
    branch_df=pd.DataFrame(groups[branch], columns=['text'])
    branch_df['category']=branch
    examine_topics(branch_df, rand_lines = 10, top_n_words = 20,s_words=s_words, show_top=show_top)

def len_group(groups,branch):
    '''
    Used in start()
    get the number of items in a group
    '''
    branch_df=pd.DataFrame(groups[branch], columns=['text'])
    branch_df['category']=branch
    return len(branch_df)

def tree_string(all_vals):
    '''
    Used in start()
    get the tree string based on all_vals
    the input all_vals should be a list of strings 
    '''
    string='\n\t'+sorted(all_vals)[0]+'\n' #first row of tree
    for i in sorted(all_vals):
        val=(re.split('[ ]',i)[0]) 
        num_spaces=int((len(val)+1)/2-1) #number of characters (which determines number of spaces)
        if num_spaces>0:
            string=string+('\t'*num_spaces+'|_______'+i+'\n\n')
    string=string+('\n')
    return string

def start():
    '''
    Main function that allows inout/output
    Adds steps to log 
    performs clustering based on researcher input
    Output summary and tree of final clusters
    '''
    print('Hello and welcome to My Voice Challenge 2021!')
    #0th input
    time.sleep(0.05)
    log_path=input('What would you like to name the log file? ')
    with open(log_path, 'a') as f: #create log file 
        #add log file beginning
        log_item=("__________________________________________________________________________")
        f.write("%s\n" % log_item)
        log_item=(datetime.now().strftime("%D-%H:%M:%S")+" Log Begin")
        f.write("%s\n" % log_item)
        f.flush()

        #while loop for first input
        while True:
            try:
                time.sleep(0.05)
                filepath = input("\nWhat is the file path for your excel file? ")
                df=format_df(filepath, 'Cleaned')
            except:
                print("\nSorry, file does not exist.")
                continue
            else:
                break
        #add excel file to log
        log_item=(datetime.now().strftime("%D-%H:%M:%S")+" Excel File: "+filepath)
        f.write("%s\n" % log_item)
        f.flush()

        #create a formatted df for clustering
        df=format_df(filepath, 'Cleaned')

        #while loop for continuing algorithm (in the end)
        cont_bool=True
        while cont_bool==True:
            clear_output()
            q=df['Question'].unique()
            print('\nThe questions are:\n')
            #for loop to print questions
            for i in range(len(q)):
                print(str(i+1)+')',q[i])

            #while loop for second input
            while True:
                try:
                    time.sleep(0.05)
                    q_num=int(input('\nWhich question would you like to explore? (Input an integer) '))
                except ValueError:
                    print("\nSorry, I didn't understand that.")
                    continue
                else:
                    break
            #add question selected to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Question selected: '+str(q_num))
            f.write("%s\n" % log_item)
            f.flush()
            # get study_id
            study_id=df[df['Question_Number']==q_num].index

            #while loop to continue with stop words
            stop_words=[]
            count1=0
            while True:
                #print top words
                print('\n')
                top_words(df.loc[df.Question_Number == q_num, "Response"].astype(str).to_list(),20,stop_words)
                #while loop for third input
                while True:
                    try:
                        time.sleep(0.05)
                        s_word = input("\nDo you want to remove some words? Note: This does not affect the clustering algorithm, it only affect what will be displayed (Input y or n) ")
                        if s_word!='y' and s_word !='n':
                            raise IOError
                    except IOError:
                        print("\nSorry, I didn't understand that.")
                        continue
                    else:
                        break
                if s_word =='y': #contine removing stop words
                    #forth input
                    count1=count1+1
                    time.sleep(0.05)
                    words=input("\nWhich words do you want to remove? (please separate words with space or comma)")
                    words=re.split('[ ,]',words)
                    #Add stop words to log
                    log_item=(datetime.now().strftime("%D-%H:%M:%S")+ " Stop Words: "+str(words))
                    f.write("%s\n" % log_item)
                    f.flush()
                    #for loop to append all stop words
                    for i in words:
                        stop_words.append(i)
                    continue
                if (s_word=='n' and count1==0): #no more stop words and first iteration
                    #write to log
                    log_item=(datetime.now().strftime("%D-%H:%M:%S")+' No Stop Words Provided')
                    f.write("%s\n" % log_item)
                    f.flush()
                    break
                else:#no more stop words and second iteration
                    #does not write to log
                    break

            #create the four initial clusters
            print('\nCreating your clusters...')
            full_df,encoded=create_initial_clusters(df,q_num)
            clear_output()
            print('\nInitial clusters created.')
            print("\nLet's examine some clusters!\n")

            #add created cluster to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+ " Initial Clusters Created")
            f.write("%s\n" % log_item)
            f.flush()

            #while loop for splitting
            complete=[]
            all_vals=[]
            all_vals.append(str(q_num))
            all_vals.append(str(q_num)+'.1')
            all_vals.append(str(q_num)+'.2')
            stop=False
            names={}
            count=2
            while(stop==False):
                time.sleep(0.05)
                count=count+1
                groups=get_groups(full_df) #get all the groups and the items (as a dictionary)
                keys=groups.keys() #get branches such as 4.1.1
                keys=[i for i in keys if i not in complete]
                decisions=[]
                #for loop for iterating through branches
                for branch in keys:
                    name=''
                    all_vals.append(branch)
                    print('Your Cluster is stored in the following tree: \n')
                    print(tree_string(all_vals))
                    #add current tree to log
                    log_item=(datetime.now().strftime("%D-%H:%M:%S")+ " Current Tree: "+tree_string(all_vals))
                    f.write("%s\n" % log_item)
                    f.flush()
                    #while loop for shuffling
                    done_shuffling=False
                    num=0
                    while(done_shuffling==False):
                        if num==0:
                            examine_branch(groups,branch,stop_words,show_top=True)
                            num=num+1
                        else:
                            examine_branch(groups,branch,stop_words,show_top=False)
                        num_items=len_group(groups,branch)
                        #while loop for fifth input (looping)
                        while True:
                            try:
                                time.sleep(0.05)
                                shuffle = input("Do you want to shuffle the sample? (Input y or n) ")
                                if shuffle!='y' and shuffle !='n':
                                    raise IOError
                            except IOError:
                                print("\nSorry, I didn't understand that.")
                                continue
                            else:
                                break
                        if shuffle=='n':
                            done_shuffling=True

                    #while loop for sixth input 
                    while True:
                        try:
                            time.sleep(0.05)
                            split = input("\nDo you want to split? (Input y or n) ")
                            if split!='y' and split !='n':
                                raise IOError
                        except IOError:
                            print("\nSorry, I didn't understand that.")
                            continue
                        else:
                            break
                    decisions.append(split)# add splitting decisions
                    if (split == 'n' or (num_items<2)): #no splitting requested or splitting not possible
                        if(split=='y' and num_items<2): #splitting not possible
                            print('\nToo few items in cluster to split')
                        #seventh input
                        time.sleep(0.05)
                        name = input("\nWhat do you want to name this cluster? ")
                        #add decision to log
                        log_item=(datetime.now().strftime("%D-%H:%M:%S")+" No Split On "+branch)
                        f.write("%s\n" % log_item)
                        f.flush()
                        #add cluster name to log
                        log_item=(datetime.now().strftime("%D-%H:%M:%S")+" Cluster Named: "+branch+", "+name)
                        f.write("%s\n" % log_item)
                        f.flush()
                        names[branch]=name
                    else: #spliting requested and splitting possible
                        #add decisions to log
                        log_item=(datetime.now().strftime("%D-%H:%M:%S")+" Splitting On "+branch)
                        f.write("%s\n" % log_item)
                        f.flush()
                    #change last item of all_vals to include name (for future trees)
                    all_vals.append(all_vals.pop()+" "+name)
                    clear_output()
                d={'y':1,'n':0} #map string to binary so we can find the sum
                decisions=[d[i] for i in decisions]
                tup=[*zip(keys,decisions)]
                to_split=[i[0] for i in tup if i[1]==1]
                clear=[i[0] for i in tup if i[1]==0]
                #for loop to add branches to complete
                for i in clear:
                    complete.append(i)
                #for loop to add branches to split
                for i in to_split:
                    try:
                        full_df = cluster(full_df, encoded = encoded, n_clusters = 2, cat = i) #try to split
                    except:
                        complete.append(i) #add cannot split to complete
                        continue
                if (sum(decisions)==0): #no more splitting requested (all decisions are 'n')
                    stop=True 
                else:
                    print('\nNext round of splitting.\n')

            #end of splitting

            #create theme column with the provided names
            full_df["Theme"] = full_df["category"].map(names)
            #add study_id
            full_df['Study_ID']=study_id
            #add splitting complete to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+ " Splitting Complete")
            f.write("%s\n" % log_item)
            f.flush()
            time.sleep(0.05)
            clear_output()
            #print summaries
            examine_topics(full_df, rand_lines = 10, top_n_words = 20,s_words=stop_words) #print all clusters
            print('\nYour clustered data has been stored in full_df with the following distribution.')
            print(dict(FreqDist(full_df["Theme"])))
            print('\nYour clusters are named according to the following dictionary')
            print(sorted(((k,v) for k,v in names.items())))
            print(' ')
            print('Your Cluster is stored in the following tree: \n')
            print(tree_string(all_vals))#print final tree

            #add distribution to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Distribution: '+ str(dict(FreqDist(full_df["Theme"]))))
            f.write("%s\n" % log_item)
            f.flush()
            #add dictionary to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Dictionary: '+ str(sorted(((k,v) for k,v in names.items()))))
            f.write("%s\n" % log_item)
            f.flush()
            #add branches to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' All branches: '+str(all_vals))
            f.write("%s\n" % log_item)
            f.flush()
            #add tree string to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Tree String: ' + tree_string(all_vals))
            f.write("%s\n" % log_item)
            f.flush()
            clear_output()
            #while loop for eighth input
            while True:
                try:
                    time.sleep(0.05)
                    save = input("\nWould you like to save the results into a csv? (Input y or n) ")
                    if save!='y' and save !='n':
                        raise IOError
                except IOError:
                    print("\nSorry, I didn't understand that.")
                    continue
                else:
                    break
            if save=='y':#CSV requested
                #while loop for ninth input
                while True:
                    try:
                        time.sleep(0.05)
                        filepath = input("\nWhat is the file path for your csv file? (ex. data/theme.csv or theme.csv) ")
                        full_df.to_csv(filepath)
                        print('\nCSV saved to {}'.format(filepath))
                    except:
                        print("\nSorry, the file path does not exist.")
                        continue
                    else:
                        break
                #add csv filepath to log
                log_item=(datetime.now().strftime("%D-%H:%M:%S")+" CSV File: "+filepath)
                f.write("%s\n" % log_item)
                f.flush()
            else:#csv not requested
                log_item=(datetime.now().strftime("%D-%H:%M:%S")+' No CSV Requested')
                f.write("%s\n" % log_item)
                f.flush()
            #while loop for last input
            while True:
                try:
                    time.sleep(0.05)
                    cont = input('\nWould you like to analyze another question? This will overwrite the current clusters. (Input y or n) ')
                    if cont!='y' and cont !='n':
                        raise IOError
                except IOError:
                    print("\nSorry, I didn't understand that.")
                    continue
                else:
                    break
            if cont == 'n': #no more questions
                clear_output()
                #print summaries (again)
                print('\nYour clustered data has been stored in full_df with the following distribution.')
                print(dict(FreqDist(full_df["Theme"])))
                print('\nYour clusters are named according to the following dictionary')
                print(sorted(((k,v) for k,v in names.items())))
                print(' ')
                print('Your Cluster is stored in the following tree: \n')
                print(tree_string(all_vals))
                print('\nThank you for clustering with us!')
                cont_bool=False
            #add full_df to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Full DataFrame:\n' + str(full_df))
            f.write("%s\n" % log_item)
            f.flush()
            #add log ending to log
            log_item=(datetime.now().strftime("%D-%H:%M:%S")+' Log complete \n\n__________________________________________________________________________')
            f.write("%s\n" % log_item)
            f.flush()
    # set index to study_id and return full_df and all_vals
    full_df=full_df.set_index('Study_ID')
    f.close()
    return (full_df,all_vals)
