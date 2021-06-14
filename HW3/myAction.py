import numpy as np
# import heapq

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )
    return actionMat

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    # holdingStack_index = np.zeros([priceMat.shape[0],1], dtype=int)
    # holdingStack_count = np.zeros([priceMat.shape[0],1], dtype=float)
    # holdingCash = np.zeros([priceMat.shape[0],1], dtype=float)
   
    # # 第一天action 起始cash=1000
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

    # # 從第二天到倒數第二天的action，最後一天的隔天沒有price資料不進行action
    # for i in range(1,priceMat.shape[0]-1):
    #     R = priceMat.shape[1]
    #     RR = 1
    #     for j in range(priceMat.shape[1]):
    #         if priceMat[i+1][j] > priceMat[i][j]:
    #             if priceMat[i+1][j]/priceMat[i][j]>RR:
    #                 RR = priceMat[i+1][j]/priceMat[i][j] #存所有股票隔天漲幅最高的數值
    #                 R = j #存所有股票隔天漲幅最高的股票index
 
    #     if RR*(1-transFeeRate) > 1:  #漲幅*交易手續還是賺 
    #         # print(RR) 
    #         if holdingStack_index[i-1][0] == priceMat.shape[1]: #前一天未持有股票
    #             actionMat.append([i, -1, R, holdingCash[i-1][0]])
    #             holdingStack_index[i][0] = R
    #             holdingStack_count[i][0] = holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][R]
    #             holdingCash[i][0] = 0
    #         elif holdingStack_index[i-1][0] != R and RR*(1-transFeeRate)*(1-transFeeRate) > priceMat[i+1][holdingStack_index[i-1][0]]/priceMat[i][holdingStack_index[i-1][0]] : #持有股票和隔天漲幅最大的股票不同 且 加上兩層手續費仍然賺
    #             actionMat.append([i, holdingStack_index[i-1][0], R, holdingStack_count[i-1][0]*priceMat[i][holdingStack_index[i-1][0]]])
    #             holdingStack_index[i][0] = R
    #             holdingStack_count[i][0] = holdingStack_count[i-1][0]*priceMat[i][holdingStack_index[i-1][0]]*(1-transFeeRate)*(1-transFeeRate)/priceMat[i][R]
    #             holdingCash[i][0] = holdingCash[i-1][0]
    #         else: #持有股票和隔天漲幅最大的股票相同 或 持有股票和隔天漲幅最大的股票不同但是加上兩層手續費不賺
    #             holdingStack_index[i][0] = holdingStack_index[i-1][0]
    #             holdingStack_count[i][0] = holdingStack_count[i-1][0]
    #             holdingCash[i][0] = holdingCash[i-1][0]      
    #     else: #隔天股票全跌或是漲幅最大的股票加上手續費就不賺了
    #         if holdingStack_index[i-1][0] != priceMat.shape[1]: #前一天持有股票
    #             if priceMat[i][holdingStack_index[i-1][0]]/priceMat[i+1][holdingStack_index[i-1][0]]*(1-transFeeRate) > 1: #如果不賣掉省手續費隔天就會賠
    #                 actionMat.append([i, holdingStack_index[i-1][0], -1, holdingStack_count[i-1][0]*priceMat[i][holdingStack_index[i-1][0]]])
    #                 holdingStack_index[i][0] = priceMat.shape[1] #沒有持有股票，股票index為4因為股票只有4隻，index為0~3
    #                 holdingStack_count[i][0] = 0 #持有的股票數為0
    #                 holdingCash[i][0] = holdingStack_count[i-1][0]*priceMat[i][holdingStack_index[i-1][0]]*(1-transFeeRate)
    #             else: #不賣掉的話考慮進手續費，不動比較賺
    #                 holdingStack_index[i][0] = holdingStack_index[i-1][0]
    #                 holdingStack_count[i][0] = holdingStack_count[i-1][0]
    #                 holdingCash[i][0] = holdingCash[i-1][0]
    #         else: #手上只有現金，目前沒有好的股票投資，所以現金保留
    #             holdingStack_index[i][0] = holdingStack_index[i-1][0]
    #             holdingStack_count[i][0] = holdingStack_count[i-1][0]
    #             holdingCash[i][0] = holdingCash[i-1][0]
    # print(actionMat)
    # print(holdingCash)
    # print(holdingStack_count)
    # print(holdingStack_index)
    # print(np.shape(actionMat))
    # a = 0
    # for i in range(992):
    #     if holdingStack_index[i][0] == 4:
    #         a +=1
    # print(a)
    # print(holdingStack)
    # print(np.shape(actionMat))
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
            list = [0, 1, 2, 3]
            list.remove(j)
            holdingStack[i+1][j] = max(holdingStack[i][j], holdingCash[i][0]*(1-transFeeRate)/priceMat[i+1][j])
            for k in list:      
                if holdingStack[i][k]*priceMat[i][k]/priceMat[i][j]*(1-transFeeRate)*(1-transFeeRate) > holdingStack[i+1][j]:
                    holdingStack[i+1][j] = holdingStack[i][k]*priceMat[i][k]/priceMat[i][j]*(1-transFeeRate)*(1-transFeeRate)            
            if holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate) > holdingCash[i+1][0]:
                holdingCash[i+1][0] = holdingStack[i][j]*priceMat[i+1][j]*(1-transFeeRate)
    # print(holdingCash)
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
    a = 0
    b = 0
    for i in range(priceMat.shape[0]-1, 0, -1):
        if index == 4:
            test1 = 4
            # test1 = np.argmax(holdingStack[i-1][0]*priceMat[i][0]*(1-transFeeRate), holdingStack[i-1][1]*priceMat[i][1]*(1-transFeeRate), holdingStack[i-1][2]*priceMat[i][2]*(1-transFeeRate), holdingStack[i-1][3]*priceMat[i][3]*(1-transFeeRate), holdingCash[i-1][0])
            step = holdingCash[i-1][0]
            for j in range(priceMat.shape[1]):
                if holdingStack[i-1][j]*priceMat[i][j]*(1-transFeeRate) > step:
                    step = holdingStack[i-1][j]*priceMat[i][j]*(1-transFeeRate)
                    test1 = j
            if test1 != 4:
                actionMat.insert(0, [i-1, test1, -1, holdingStack[i-1][test1]*priceMat[i-1][test1]])
                index = test1
            a += 1
        else :
            # test2 = np.argmax(holdingStack[i-1][index], holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index])
            list = [0, 1, 2, 3]
            list.remove(index)
            # test2 = holdingStack[i][index]
            # test3 = index
            # for k in list:
            #     if holdingStack[i-1][k]*priceMat[i-1][k]/priceMat[i-1][index]*(1-transFeeRate)*(1-transFeeRate) == test2:
            test3 = index
            test2 = holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]
            for j in list:
                if holdingStack[i-1][j]*priceMat[i-1][j]/priceMat[i-1][index]*(1-transFeeRate)*(1-transFeeRate) > test2:
                    test2 = holdingStack[i-1][j]*priceMat[i-1][j]/priceMat[i-1][index]*(1-transFeeRate)*(1-transFeeRate) 
                    test3 = j
            if test3 != index and test2 > holdingStack[i-1][index]:
                actionMat.insert(0, [i-1, test3, index, holdingStack[i-1][test3]*priceMat[i-1][test3]])
                index = test3
            else:
                if holdingStack[i-1][index] < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
                    actionMat.insert(0, [i-1, -1, index, holdingCash[i-1][0]]) 
                    index = 4
            #         test3 = k
            # if test3 != index and test2 > holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
            #     actionMat.insert(0, [i, test3, index, holdingStack[i-1][test3]*priceMat[i-1][test3]])
            #     index = test3
            # elif test3 == index and test2 < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
            #     actionMat.insert(0, [i, -1, index, holdingCash[i-1][0]]) 
            #     index = 4
            # else:
            #     pass
            # if holdingStack[i-1][index] < holdingCash[i-1][0]*(1-transFeeRate)/priceMat[i][index]:
            #     actionMat.insert(0, [i, -1, index, holdingCash[i-1][0]]) 
            #     index = 4
            b += 1
    # print(actionMat)
    # print(373453655.9985885*(1-transFeeRate)*(1-transFeeRate)*priceMat[989][0]/priceMat[986][0], holdingStack[991][3]*priceMat[991][3])
    # print(holdingStack[991][3]*priceMat[991][3], holdingCash[991][0], 111133856*(1-transFeeRate)*priceMat[991][3]/priceMat[989][3])
    # print(a,b,a+b)
    return actionMat


# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
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
                holdingStack_index[i+1][0] = priceMat.shape[1]
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

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
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
    # print(holdingCash[991][0])

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
    if index != 4:
        actionMat.insert(0, [large_k+K, -1, index, holdingCash[large_k+K-1][0]])
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