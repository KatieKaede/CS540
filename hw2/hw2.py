#!/usr/bin/env python
# coding: utf-8

# In[110]:


import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    e = [0] * 26  # English letter probabilities
    s = [0] * 26  # Spanish letter probabilities

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            e[ord(char) - ord('A')] = float(prob)

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)

    return e, s


def shred(filename):
    '''
    Q1 After the input file is processed, this creates a list from A-Z for our individual counts
    
    '''
    X = [0] * 26

    with open(filename, encoding='utf-8') as f:
        for line in f:
            for char in line.upper():
                if 'A' <= char <= 'Z':
                    X[ord(char) - ord('A')] += 1
    return X


def log_probabilities(X, e, s):
    '''
    Q2 Calculates Eng/Spanish based on Bayes theorem
    
    X is the 26 letters
    e is the 26 probabilities of each letter for English
    s is the 26 probabilities of each letter for Spanish
    
    '''
    log_e = 0
    log_s = 0

    for i in range(26):
        count = X[i]
        if count > 0:
            log_e_i = math.log(e[i]) if e[i] > 0 else 0
            log_s_i = math.log(s[i]) if s[i] > 0 else 0
            
            # for the accumulated sum of found letter probabilities add it all up!
            log_e += count * log_e_i
            log_s += count * log_s_i

    return log_e, log_s


def f_values(log_e, log_s, prior_english, prior_spanish):
    '''
    Q3 This function creates F(English) and F(Spanish)
    
    '''
    F_english = math.log(prior_english) + log_e
    F_spanish = math.log(prior_spanish) + log_s
    return F_english, F_spanish


def compute_p_english(F_english, F_spanish):
    '''
    Q4 This function computes the probability P(Y=English | X) based on subtracting F(English) and F(Spanish)
    
    '''
    diff = F_spanish - F_english

    if diff >= 100:
        return 0.0000
    elif diff <= -100:
        return 1.0000
    else:
        return 1 / (1 + math.exp(diff))

if __name__ == "__main__":
    letter_file = sys.argv[1]
    prior_english = float(sys.argv[2])
    prior_spanish = float(sys.argv[3])

    e, s = get_parameter_vectors()

    X = shred(letter_file)

    # Print Q1 letter frequencies
    print("Q1")
    for i in range(26):
        print(chr(i + ord('A')), X[i])

    # Print Q2 probabilities based on vector A
    X1 = X[0]
    log_e1 = math.log(e[0]) if e[0] > 0 else 0
    log_s1 = math.log(s[0]) if s[0] > 0 else 0

    result_e1 = X1 * log_e1
    result_s1 = X1 * log_s1
    
    # Print in format
    print("Q2")
    print(f"{result_e1:.4f}")
    print(f"{result_s1:.4f}")
    
    # call this one more time for Q3/Q4
    log_e, log_s = log_probabilities(X, e, s)

    # Compute F(English) and F(Spanish) for Q3
    F_english, F_spanish = f_values(log_e, log_s, prior_english, prior_spanish)
    print("Q3")
    print(f"{F_english:.4f}")
    print(f"{F_spanish:.4f}")

    # Compute P(Y = English | X) for Q4
    p_english = compute_p_english(F_english, F_spanish)
    print("Q4")
    print(f"{p_english:.4f}")


# In[ ]:





# In[ ]:




