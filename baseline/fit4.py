#!python 
"""
   This version differs from fit3E.py in that it 
   looks at several different models to predict the individual choices.
   It computes a synset_polarity profile for each example, and 
   trains a classification model for each profile.

   My intention is to use the sense-polarity tests on prev & extra
   to partition the training into 9 different model families


     
"""
import json
import math
import numpy as np
import numpy.linalg as nl
import os
import random
import scipy.stats as st
import sklearn.linear_model as sl
import sklearn.preprocessing as pr
import sys
import train4

VERSION = 4
#T_TESTS = 12 # this might be wrong now.
#T_DAT_SIZE = 2280 # this is now computed in main

# TINY UI
prediction_file = 'predictions4.json'
TEST_INPUT_FILE = '../tfile' # '../dev.json'
TEST_REGIME = True # if processing a TEST, not a DEV file
POLYNOMIAL = False #True
SCALE_INPUT = True
GNUPLOT = True
APPROX_SCORES = 'NONE' # fractional scores now allowed
STUMBLE_ON = True  # include cases with ENDING datavalues in dev output
DEV_OUTPUT = 'dev3E.bin'  # set to None to avoid this output
Train_output = f't{VERSION}.bin'

if len(sys.argv) > 1:
    TEST_INPUT_FILE = sys.argv[1]

if len(sys.argv) > 2:
    Train_output = sys.argv[2]
# currently imagine following values for APPROX_SCORES:
# 'NONE'  # go ahead and use fractional scores

# 'ROUND' #round scores up or down

# 'RAND'  # assign integer scores randomly based on estimated std deviation.
       # that is, given the mean estimate and stddev estimate, 
       # 1) assign probabilities to each of the integer possibilities i 
       #   (corresponding to the probability that a normal sample would fall
       # between i-.5 and i_.5)
       # 2) normalize the sum of the probabilities to 1.0  (sum will be less, of
       # course, because normal distribution would allow for scores > 5 or  < 1
       # 3) arrange the probabilities as a cumulative series, for example
       #   1. is [0, P1), 2. is [P1, P1+P2), 3. is [P1+P2, P1+P2+P3), etc.
       # 4) take a random number in the interval 0-1
       #    and choose the integer whose range includes that random number.



#This array is now initialized based on the file prolog
# and may be resized if POLYNOMIAL_FEATURES is used:
#T = np.ndarray((T_DAT_SIZE,T_TESTS), np.float32)

def main():
    global A, S, T

    np.seterr(all='raise')  # assist pinpointing problems

    if DEV_OUTPUT is not None:
        DEV_fo = open(DEV_OUTPUT,'wb')
        # next, copy out prolog

    with open(Train_output,'rb') as fib:
        fib_size = os.fstat(fib.fileno()).st_size # get file size for later
        P0,P1,P2,P3 = fib.read(4) # byte order issues?
        if P0 != VERSION:  # is this file version correct?
            print(f'?file version is not {VERSION}, but {P0}')
            harrumph()
        if DEV_OUTPUT is not None:
            writeB(DEV_fo, P0)
            writeB(DEV_fo, P1)
            writeB(DEV_fo, P2)
            writeB(DEV_fo, P3)
        #print (f'{P1} Components')
        ind = 0
        for _ in range(P1):
            Ckey = fib.read(8)
            if DEV_OUTPUT is not None:
                DEV_fo.write( Ckey)
            #print (f' {ind}.  {Ckey}')
            ind += 1
        #print(f'\n{P2} Synthetic components')
        for _ in range(P2):
            S0,S1,S2,S3 = fib.read(4)
            if DEV_OUTPUT is not None:
                writeB(DEV_fo, S0)
                writeB(DEV_fo, S1)
                writeB(DEV_fo, S2)
                writeB(DEV_fo, S3)
            #print(f' {ind}.  Action[{S0}]({S1},{S2},{S3})')
            ind += 1
        #print(f'\n{P3} Tests')
        Polarity_tests = []
        first_test = 0 # position of the first test in the t3.bin record.
        for t_ in range(P3):
            S0,S1,S2,S3 = fib.read(4)
            if DEV_OUTPUT is not None:
                writeB(DEV_fo, S0)
                writeB(DEV_fo, S1)
                writeB(DEV_fo, S2)
                writeB(DEV_fo, S3)
            #print(f'         Action[{S0}]({S1},{S2},{S3})')
            if S0 == 17: #Synset_Polarity action
                Polarity_tests.append((first_test+t_,S0,S1,S2,S3))
        #That should finish the preamble
        Preamble_size = 4*(1+2*int(P1)+int(P2)+int(P3))
        Record_size = 4*(3+int(P3))
        T_DAT_SIZE = train_records = (fib_size - Preamble_size)//Record_size
        if T_DAT_SIZE * Record_size + Preamble_size != fib_size:
            print('Maybe I misunderstand the file sizing')
            scream()

        # now read the training data record by record

        rec_num = 0
        T = np.ndarray((T_DAT_SIZE,P3), np.float32)
        PP = np.zeros(T_DAT_SIZE, dtype=np.int8) # polarity profile
        CE = np.ndarray((T_DAT_SIZE,5), np.int8)
        A = np.ndarray(T_DAT_SIZE, np.float32)
        S = np.ndarray(T_DAT_SIZE, np.float32)

        Record_size = 4*(3+P3)
        polarity_profiles = set()
        while True:

            record = fib.read(Record_size)

            if len(record) < Record_size: 
                print(f'Reached EOF of t{VERSION}.bin after reading {rec_num} records')
                if len(record) !=0:
                    print (f'last partial record contained {len(record)} bytes')
                break
            
            line = np.frombuffer(record[0:4], np.int32)[0]

            # version 4 difference:
            compr_ch = np.frombuffer(record[4:8], np.int32)[0]
            #A[rec_num] = np.frombuffer(record[4:8], np.float32)[0]
            CE[rec_num] = train4.uncompress_choices(compr_ch)
            A[rec_num] = CE[rec_num].mean()

            S[rec_num] = np.frombuffer(record[8:12], np.float32)[0]
            Te = np.frombuffer(record[12:132], np.float32)
            T[rec_num] = Te # was: T[rec_num,1:] = Te
            temp = compute_polarity(Te,Polarity_tests)
            PP[rec_num] = temp
            polarity_profiles.add(temp)
            
            rec_num += 1



    #Now have read the results of the tests on the training data into
    # A, S, T;   so perform regression:
    
    #first try some preprocessing:
    if POLYNOMIAL:
        poly = pr.PolynomialFeatures(1)
        T = poly.fit_transform(T)

    if SCALE_INPUT:
        scaler = pr.StandardScaler().fit(T)
        T = scaler.transform(T)
    


    # now train models for each polarity profile
    #CE = np.load('trainCE.bin')
    modelsD = dict()  #modelsD dict replaces an early models list
    r_sq_ls = []
    for p in polarity_profiles:
        F = PP[:] == p

        Tp = T[F,:]
        Cp = CE[F,:5]

        #noe_modelCs =sl.MultiTaskLasso(alpha=0.5).fit(Tnoe,Cnoe[:,i])
        #noe_modelCsX =sl.ElasticNet(alpha=0.5).fit(Tnoe,Cnoe[:,i])

        _mods = []
        for i in range(5):
            _mods.append(sl.LogisticRegression(max_iter=1000, C=0.99)
                              .fit(Tp,CE[F,i]))
            r_sq_ls .append( (p,i,"%4.2f"%_mods[i].score(Tp,CE[F,i])) )

        modelsD[p] = _mods

        #reg_modelAS = sl.MultiTaskLasso(alpha=0.5).fit(T,AS)
        #EN = reg_modelAS.intercept_
        #E = reg_modelS.coef_
        #as_r_sq = reg_modelAS.score(T,AS)

       # print('CN = ', CN, 'C =',C, 'R^2=',a_r_sq)
       # print('DN = ', DN, 'D =', D, 'R^2=',s_r_sq)
        #print('EN = ', EN, 'E =', E, 'R^2=',r_sq_ls)
    print( 'Train R^2=',r_sq_ls)

    with open(TEST_INPUT_FILE) as fi:
        dev = json.load(fi)
        print (f'length development file: {len(dev)}')

    Tshape = T.shape

                
    # We are done with T, so we could deallocate memory
    #A = S = T = None

    # try an easier development set also?
    #if not STUMBLE_ON:
    #    CE = F = Oce = None
    #    CE = np.load('devCE.bin')
    #    F = CE[:,5]==0
    #    Oce = CE[F,:5]
    #    ldevCh = Oce.shape[0]
    #else:
    #    # iterate through the keys in dev file?
    ldevCh = len(dev)

    #Tnoe = T[F,:]
    

    # in case we used PolynomialFeatures, T may have changed shape since
    # original allocation
    #VT = np.ndarray(shape=(len(dev),Tshape[1]), dtype=np.float32)
    #VAS = np.ndarray(shape=(len(dev),2), dtype=np.float32)
    #VASp = np.ndarray(shape=(len(dev),2), dtype=np.float32)
    VT = np.ndarray(shape=(ldevCh,Tshape[1]), dtype=np.float32)
    VAS = np.ndarray(shape=(ldevCh,2), dtype=np.float32)
    VASp = np.ndarray(shape=(ldevCh,2), dtype=np.float32)
    VCh = np.ndarray(shape=(ldevCh,5), dtype=np.uint8) # to predict into
    Key = np.ndarray(shape=(ldevCh), dtype=np.int32) 
    VPP = np.ndarray(shape=(ldevCh), dtype=np.int8) #synset Polarity Profile

    # allocate a Tf file, with a one in the right place (Te still hanging around
    Tf = np.ndarray(shape = (1,Te.shape[0]), dtype = np.float32)
    #Tf[0,0] = 1.0  # this isn't useful for LinearRegression.fit()

    av_err_sum = 0
    st_err_sum = 0
    dev_cases = 0



    # build VT, VCh, VAS files for development data
    for i, (key, example) in enumerate(dev.items()):
            
            j = dev_cases

            av, st, Te = train4.evalu(example, key)
            if not TEST_REGIME:
                ce = train4.choice_list_standardize(example['choices'])
                CE[dev_cases] = ce #train4.uncompress_choices(ce)

            this_polarity = compute_polarity(Te,Polarity_tests)

            Tf[0] = Te # copy test values into Tf # Tf[0,1:] = Te # copy test values into Tf
            
            #do any preprocessing:
            if POLYNOMIAL:
                Tg = poly.fit_transform(Tf)
                if SCALE_INPUT:
                    Tg = scaler.transform(Tg)

            elif SCALE_INPUT:
                Tg = scaler.transform(Tf)

            else: Tg = Tf

            VT[dev_cases] = Tg
            if not TEST_REGIME:
                VAS[dev_cases,0] = av
                VAS[dev_cases,1] = st
            else:
                VAS[dev_cases,0] = 0
                VAS[dev_cases,1] = 0
            VPP[dev_cases] = this_polarity
            #choices = example['choices']
            #for j in range(5):
            #    VCh[dev_cases,j] = choices[j] # we want to predict these...
            Key[dev_cases] = int(key) # .append(key)
            dev_cases += 1
            
    pp_counts = dict()
    # now do a little computation on the data to predict the estimates
    SPP = np.argsort(VPP, axis=0)  
    for p in polarity_profiles:
        F = VPP[SPP] == p
        tmp = pp_counts[p] = F.sum()
        if tmp == 0:  # nothing to predict.  Since |dev|<|train| not strange.
            continue  # presumably opposite could also occur; no prediction
                      # for some values of p which don't appear in training.
        for i in range(5):
                TEMP = modelsD[p][i].predict(VT[SPP[F]])
                VCh[SPP[F],i] = TEMP

        # now compute average and standard deviation
        TEMP = VCh[SPP[F]]
        TEMP2 = TEMP ** 2
        TEMP = np.mean(TEMP, axis=1)
        TEMP2 = np.mean(TEMP2, axis = 1)
        TEMP3 = np.sqrt(TEMP2-TEMP**2)
        if np.isnan(TEMP .dot (TEMP)):
            print ('nan')
        VASp[:,0][SPP[F]] = TEMP
        VASp[:,1][SPP[F]] = TEMP3
        if np.isnan(VASp[140,0]):
             pass

    VCh.sort(axis=1)  # this sort is probably unnecessary.  But harmless


    #with calculations finished*, and hopefully stored in correct order...
    #  * if the test set includes an unmodeled polarity, it's not calculated
    #now output the results to the prediction file

    if GNUPLOT:
        GNU_fo = open('gnuplot.dat','w')
        print(f'# key, A, S, pA, pS, lastTest', file=GNU_fo)
            
    with open(prediction_file,'w') as fo:
                 
        for i in range(ldevCh):
            if GNUPLOT: 
                print(f' "{Key[i]}", {example["average"]}, {example["stdev"]}, {VASp[i,0]}, {VASp[i,1]}, {VT[i:-1]}', file = GNU_fo)
            prediction = build_int(VASp[i,0], VASp[i,1])
            if prediction < 1 or prediction > 5:
                print(f'?Crazy prediction of {prediction} for key: {Key[i]}'
                      f' based on {VASp[i,0]} and {VASp[i,1]}')

            #if i == 0:     #break up the lines...
            #    print('[', file=fo)
            #else:
            #    print(',', file=fo)

            if APPROX_SCORES == 'NONE' :
                prediction = "%2.1F"%prediction # round to one digit fraction
            print('{'+f' "id": "{Key[i]}", "prediction": {prediction} '+'}',
                  file= fo)

            if not TEST_REGIME:
                av_err_sum += (av - VASp[i,0])**2
                st_err_sum += (st - VASp[i,1])**2
            else:
                av_err_sum = st_err_sum = 0

            if DEV_OUTPUT is not None:
                DEV_fo.write(np.int32(Key[i]))
                if not TEST_REGIME:
                    Cstuff = train4.compressed_choices(CE[i])
                else:
                    Cstuff = 0
                Dstuff = train4.compressed_choices(VCh[i])
                DEV_fo.write(np.int32(Cstuff))  # save data instead of av
                DEV_fo.write(np.int32(Dstuff))  # save predictions instead of st

                DEV_fo.write(Tg) # write out the possibly scaled test values

        # end of for loop.  write last ']'
        #print ('\n]', file=fo) # print volunteers a newline at the end, as well



    
    print(ldevCh,'dev cases.  mean-sq-error scores:', 
                                 av_err_sum/ldevCh, st_err_sum/ldevCh)

    # add the all-at-once code to compare:
    #VASq = reg_modelAS.predict(VT)
    #Vadiff = np.abs(VASp[:,0]-VASq[:,0])
    #Vsdiff = np.abs(VASp[:,1]-VASq[:,1])
    #total_a_diff = np.sum(Vadiff)
    #total_s_diff = np.sum(Vsdiff)

    R2 = []
    # this code probably doesn't work anymore:
    for p in polarity_profiles:
        modl = modelsD[p]
        R2.append((p,int(pp_counts[p])))
        for j,mod in enumerate(modl):
            R2.append('%4.3f'%mod.score(VT,VCh[:,j]))
    print('unweighted model R2 ',R2)
    #print (f'Total A difference: {total_a_diff}\n Total S diff: {total_s_diff}\n over {cases} cases')

def writeB(fob, byt):
    fob.write(np.int8(byt))

def build_int(f, stdev):
    """
    given the global APPROX_SCORES guidance,
    the predicted mean f for this example,
    and the predicted stdev, build an integer result for scoring.
    """
    if APPROX_SCORES == "NONE":
        return f
    elif APPROX_SCORES == 'ROUND':
        return int(math.floor(f+0.5))
    elif APPROX_SCORES == 'RAND':
        
        below  = st.norm.cdf((0.5-f)/stdev) 
        above = 1 - st.norm.cdf((5.5-f)/stdev)
        outside = above + below
        adjustment = 1/(1-outside)

        tops = [0]*6 # top z-score for subscript
        for i in range(1,6):
            zscore = (i + 0.5 - f)/stdev
            tops[i] = st.norm.cdf(zscore) * adjustment

        r = random.random()   # a float in the range (0,1)
        for i in range(1,5):
            if r < tops[i]:
                return i
        return 5

def compute_polarity(Te,polarity_tests):
    test1index = polarity_tests[0][0]
    #if len(polarity_tests) == 3:
    test2index = polarity_tests[-1][0]
    t1 = int(Te[test1index])
    if t1 != -1 and t1 != 0 and t1 != +1:
        complain()
    t2 = int(Te[test2index])
    if t2 != -1 and t2 != 0 and t2 != +1:
        complain()
    retval = t1 + 3*t2
    return retval
    



if __name__ == '__main__':
    main()






