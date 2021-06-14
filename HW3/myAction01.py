import numpy as np
import heapq

def myAction011(priceMat, transFeeRate):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    holdingStack = np.zeros([priceMat.shape[0], priceMat.shape[1]], dtype=float)
    # holdingStack_count = np.zeros([priceMat.shape[0],1], dtype=float)
    holdingCash = np.zeros([priceMat.shape[0],1], dtype=float)
   
    # 第一天action 起始cash=1000
    # RR = 1
    # R = priceMat.shape[1]
    # for j in range(priceMat.shape[1]):
    #     if priceMat[1][j] > priceMat[0][j]:
    #         if priceMat[1][j]/priceMat[0][j]>RR:
    #             RR = priceMat[1][j]/priceMat[0][j] #存所有股票隔天漲幅最高的數值
    #             R = j #存所有股票隔天漲幅最高的股票index
    # if RR*(1-transFeeRate) > 1:
    #     actionMat.append([0, -1, R, 1000])
    #     holdingStack_index[0][0] = R
    #     holdingStack_count[0][0] = 1000*(1-transFeeRate)/priceMat[0][R]
    #     holdingCash[0][0] = 0
    # else :
    #     holdingStack_index[0][0] = priceMat.shape[1]
    #     holdingStack_count[0][0] = 0
    #     holdingCash[0][0] = 1000
    for j in range(priceMat.shape[1]):
        holdingStack[0][j] = 0
    holdingCash[0][0] = 1000

    # 從第二天到倒數第二天的action，最後一天的隔天沒有price資料不進行action
    for i in range(0,priceMat.shape[0]-1):
        holdingCash[i+1][0] = holdingCash[i][0]
        for j in range(priceMat.shape[1]):
            holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])            
            if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
    # print(holdingCash[991][0])
    # print(np.argmax(holdingCash[991][0], holdingStack[991][0]*priceMat[991][0], holdingStack[991][1]*priceMat[991][1], holdingStack[991][2]*priceMat[991][2], holdingStack[991][3]*priceMat[991][3]))
    # final = np.argmax(holdingStack[991][0]*priceMat[991][0], holdingStack[991][1]*priceMat[991][1], holdingStack[991][2]*priceMat[991][2], holdingStack[991][3]*priceMat[991][3], holdingCash[991][0])
    final = 4
    test = holdingCash[991][0]
    for j in range(priceMat.shape[1]):
        if holdingStack[991][j]*priceMat[991][j] > test:
            test = holdingStack[991][j]*priceMat[991][j]
            final = j
    # print(final)
    index = final
    for i in range(priceMat.shape[0]-1, 0, -1):
        if index == 4:
            test1 = 4
            # test1 = np.argmax(holdingStack[i-1][0]*priceMat[i][0]*(1-transFeeRate), holdingStack[i-1][1]*priceMat[i][1]*(1-transFeeRate), holdingStack[i-1][2]*priceMat[i][2]*(1-transFeeRate), holdingStack[i-1][3]*priceMat[i][3]*(1-transFeeRate), holdingCash[i-1][0])
            step = holdingCash[i-1][0]
            for j in range(priceMat.shape[1]):
                if holdingStack[i-1][j]*priceMat[i][j] > step:
                    step = holdingStack[i-1][j]*priceMat[i][j]
                    test1 = j
            if test1 != 4:
                actionMat.insert(0, [i, test1, -1, holdingStack[i-1][test1]*priceMat[i][test1]])
            index = test1
        else :
            # test2 = np.argmax(holdingStack[i-1][index], holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index])
            if holdingStack[i-1][index] < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
                actionMat.insert(0, [i, -1, index, holdingCash[i-1][0]]) 
                index = 4
    # print(actionMat)
    return actionMat

def myAction022(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    holdingStack_index = np.zeros([priceMat.shape[0], priceMat.shape[1]], dtype=int)
    holdingStack_count = np.zeros([priceMat.shape[0], priceMat.shape[1]], dtype=float)
    holdingCash = np.zeros([priceMat.shape[0],1], dtype=float)
    
    largestrr_day = np.zeros(priceMat.shape[0]-1, dtype=float)
    for i in range(priceMat.shape[0]-1):
        largestrr_day[i] = max(priceMat[i+1][0]/priceMat[i][0], priceMat[i+1][1]/priceMat[i][1], priceMat[i+1][2]/priceMat[i][2], priceMat[i+1][3]/priceMat[i][3])
    ss = largestrr_day.argsort()[::1][0:K]
    holdingCash[0][0] = 1000
    # print(ss, len(ss))
    for i in range(priceMat.shape[0]-1):
        if i in ss:
            if holdingCash[i][0] == 0:
                actionMat.append([i, holdingStack_index[i][0], -1, holdingStack_count[i][0]*priceMat[i][holdingStack_index[i][0]]]) 
                holdingStack_index[i+1][0] = 4
                holdingStack_count[i+1][0] = 0
                holdingCash[i+1][0] = holdingStack_count[i][0]*priceMat[i][holdingStack_index[i][0]]*(1-transFeeRate)
            else :
                holdingStack_index[i+1][0] = holdingStack_index[i][0]
                holdingStack_count[i+1][0] = holdingStack_count[i][0]
                holdingCash[i+1][0] = holdingCash[i][0]
        else:
            if holdingCash[i][0] == 0:
                list = [0, 1, 2, 3]
                k = holdingStack_index[i][0]
                list.remove(k)
                big = priceMat[i+1][k]/priceMat[i][k]
                big_index = k
                for j in list:
                    if priceMat[i+1][j]/priceMat[i][j]*(1-transFeeRate)*(1-transFeeRate) > big:
                        big_index = j 
                        big = priceMat[i+1][j]/priceMat[i][j]*(1-transFeeRate)*(1-transFeeRate)                    
                if big_index != k:
                    actionMat.append([i, k, big_index, holdingStack_count[i][0]*priceMat[i][k]])
                    holdingStack_index[i+1][0] = big_index
                    holdingStack_count[i+1][0] = holdingStack_count[i][0]*priceMat[i][k]*(1-transFeeRate)*(1-transFeeRate)/priceMat[i][big_index]
                    holdingCash[i+1][0] = holdingCash[i][0]
                else:
                    holdingStack_index[i+1][0] = holdingStack_index[i][0]
                    holdingStack_count[i+1][0] = holdingStack_count[i][0]
                    holdingCash[i+1][0] = holdingCash[i][0]
            else:
                # big_index = np.argmax(priceMat[i+1][0]/priceMat[i][0], priceMat[i+1][1]/priceMat[i][1], priceMat[i+1][2]/priceMat[i][2], priceMat[i+1][3]/priceMat[i][3])   
                big_index = 0
                big = priceMat[i+1][0]/priceMat[i][0]
                for j in range(1,priceMat.shape[1]):
                    if priceMat[i+1][j]/priceMat[i][j] > big:
                        big_index = j
                        big = priceMat[i+1][j]/priceMat[i][j]
                actionMat.append([i, -1, big_index, holdingCash[i][0]])
                holdingStack_index[i+1][0] = big_index
                holdingStack_count[i+1][0] = holdingCash[i][0]*(1-transFeeRate)/priceMat[i][big_index]
                holdingCash[i+1][0] = 0
    return actionMat

def myAction033(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    holdingStack = np.zeros([priceMat.shape[0], priceMat.shape[1]], dtype=float)
    holdingCash = np.zeros([priceMat.shape[0],1], dtype=float)
   
   
    # for j in range(priceMat.shape[1]):
    #     holdingStack[0][j] = 0
    holdingCash[0][0] = 1000

    # 從第二天到倒數第二天的action，最後一天的隔天沒有price資料不進行action
    large_holdingCash = 0
    large_k = 0
    for k in range(0,priceMat.shape[0]-K):
        for i in range(0,k):
            holdingCash[i+1][0] = holdingCash[i][0]
            for j in range(priceMat.shape[1]):
                holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])            
                if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                    holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
        for i in range(k,k+K):
            for j in range(priceMat.shape[1]):
                holdingStack[i+1][j] = holdingStack[i][j]
            holdingCash[i+1][0] = holdingCash[i][0]
        for i in range(k+K,priceMat.shape[0]-1):
            holdingCash[i+1][0] = holdingCash[i][0]
            for j in range(priceMat.shape[1]):
                holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])            
                if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                    holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
        if holdingCash[991][0] > large_holdingCash:
            large_holdingCash = holdingCash[991][0]
            large_k = k
    for i in range(0,large_k):
        holdingCash[i+1][0] = holdingCash[i][0]
        for j in range(priceMat.shape[1]):
            holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])            
            if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
    for i in range(large_k,k+K):
        for j in range(priceMat.shape[1]):
            holdingStack[i+1][j] = holdingStack[i][j]
            holdingCash[i+1][0] = holdingCash[i][0]
    for i in range(large_k+K,priceMat.shape[0]-1):
        holdingCash[i+1][0] = holdingCash[i][0]
        for j in range(priceMat.shape[1]):
            holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])            
            if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
    print(holdingCash[991][0])

    # print(holdingCash)
    # print(np.argmax(holdingCash[991][0], holdingStack[991][0]*priceMat[991][0], holdingStack[991][1]*priceMat[991][1], holdingStack[991][2]*priceMat[991][2], holdingStack[991][3]*priceMat[991][3]))
    # final = np.argmax(holdingStack[991][0]*priceMat[991][0], holdingStack[991][1]*priceMat[991][1], holdingStack[991][2]*priceMat[991][2], holdingStack[991][3]*priceMat[991][3], holdingCash[991][0])
    final = 4
    test = holdingCash[991][0]
    for j in range(priceMat.shape[1]):
        if holdingStack[991][j]*priceMat[991][j] > test:
            test = holdingStack[991][j]*priceMat[991][j]
            final = j
    index = final
    for i in range(priceMat.shape[0]-1,large_k+K+1, -1):
        if index == 4:
            test1 = 4
            step = holdingCash[i-1][0]
            for j in range(priceMat.shape[1]):
                if holdingStack[i-1][j]*priceMat[i][j] > step:
                    step = holdingStack[i-1][j]*priceMat[i][j]
                    test1 = j
            if test1 != 4:
                actionMat.insert(0, [i, test1, -1, holdingStack[i-1][test1]*priceMat[i][test1]])
            index = test1
        else :
                # test2 = np.argmax(holdingStack[i-1][index], holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index])
            if holdingStack[i-1][index] < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
                actionMat.insert(0, [i, -1, index, holdingCash[i-1][0]]) 
                index = 4
        # print(actionMat)
    for i in range(large_k, 0, -1):
        if index == 4:
            test1 = 4
                # test1 = np.argmax(holdingStack[i-1][0]*priceMat[i][0]*(1-transFeeRate), holdingStack[i-1][1]*priceMat[i][1]*(1-transFeeRate), holdingStack[i-1][2]*priceMat[i][2]*(1-transFeeRate), holdingStack[i-1][3]*priceMat[i][3]*(1-transFeeRate), holdingCash[i-1][0])
            step = holdingCash[i-1][0]
            for j in range(priceMat.shape[1]):
                if holdingStack[i-1][j]*priceMat[i][j] > step:
                    step = holdingStack[i-1][j]*priceMat[i][j]
                    test1 = j
            if test1 != 4:
                actionMat.insert(0, [i, test1, -1, holdingStack[i-1][test1]*priceMat[i][test1]])
            index = test1
        else :
                # test2 = np.argmax(holdingStack[i-1][index], holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index])
            if holdingStack[i-1][index] < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
                actionMat.insert(0, [i, -1, index, holdingCash[i-1][0]]) 
                index = 4
    # print(actionMat)
    # print(large_holdingCash, large_k)
    return actionMat