from sklearn import svm
import numpy as np
import dataPro as dp
import utils
import os

class svmPredict():
    def __init__(self,data = None):
        #self.filePath = filePath
        self.data = data

    def checkData(self):
        #检查数据是否完整
        data = self.data
        for i in range(len(data['open'])):
            if data['open'][i] == "nan":
                pass

    def getTrainData(self):
        #以5天的数据作为训练集，后面第三天的波动作为标签
        data = self.data
        day_step = 5
        future_d = 10
        trainRatio = 0.5
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        trainDays = int(len(data['open'])*trainRatio)
        for i in range(trainDays):
            x = []
            for j in range(day_step):
                x.append(data['open'][i+j])
                x.append(data['close'][i+j])
                x.append(data['high'][i+j])
                x.append(data['low'][i+j])
                x.append(data['close'][i+j] - data['open'][i+j])
                #x.append(np.log10(data['amount'][i+j]))
                #x.append(data['turn'][i+j])
            x_train.append(x)
            y = 100 * (data['close'][i+day_step+future_d] - data['close'][day_step+i])/data['close'][day_step+i]
            y_train.append(y)

        for i in range(trainDays,len(data['open']) - day_step - future_d):
            x = []
            for j in range(day_step):
                x.append(data['open'][i + j])
                x.append(data['close'][i + j])
                x.append(data['high'][i + j])
                x.append(data['low'][i + j])
                x.append(data['close'][i + j] - data['open'][i + j])
                #x.append(np.log10(data['amount'][i + j]))
                #x.append(data['turn'][i + j])
            x_test.append(x)
            y = 100 * (data['close'][i+day_step+future_d] - data['close'][day_step+i])/data['close'][day_step+i]
            y_test.append(y)

        return x_train,y_train,x_test,y_test

    def getTrainData_ma(self):
        #以5天的数据作为训练集，后面第三天的波动作为标签
        data = self.data
        day_step = 10
        ma = utils.MA(self.data,day_step)
        future_d = 5
        trainRatio = 0.5
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        trainDays = int(len(data['open'])*trainRatio)
        for i in range(trainDays):
            x = []
            for j in range(day_step):
                x.append(data['open'][i+j] - ma[i])
                x.append(data['close'][i+j] - ma[i])
                x.append(data['high'][i+j] - ma[i])
                x.append(data['low'][i+j] - ma[i])
                x.append(data['close'][i+j] - data['open'][i+j])
                #x.append(np.log10(data['amount'][i+j]))
                #x.append(data['turn'][i+j])
            x_train.append(x)
            y = 100 * (data['close'][i+day_step+future_d] - data['close'][day_step+i])/data['close'][day_step+i]
            y_train.append(y)

        for i in range(trainDays,len(data['open']) - day_step - future_d):
            x = []
            for j in range(day_step):
                x.append(data['open'][i + j] - ma[i])
                x.append(data['close'][i + j] - ma[i])
                x.append(data['high'][i + j] - ma[i])
                x.append(data['low'][i + j] - ma[i])
                x.append(data['close'][i + j] - data['open'][i + j])
                #x.append(np.log10(data['amount'][i + j]))
                #x.append(data['turn'][i + j])
            x_test.append(x)
            y = 100 * (data['close'][i+day_step+future_d] - data['close'][day_step+i])/data['close'][day_step+i]
            y_test.append(y)

        return x_train,y_train,x_test,y_test

    def getFlags(self):
        data = self.data
        x_train, y_train, x_test, y_test = self.getTrainData()
        flag_list = np.zeros(len(data['open']))
        clf = svm.SVR()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        for i in range(len(y_pred)):
            if y_pred[i]>0.2:
                flag_list[i+len(y_train)] = 1
            elif y_pred[i]<-0.2:
                flag_list[i+len(y_train)] = -1
            else:
                flag_list[i + len(y_train)] = 0

        return flag_list

class svmPredict_continuously(dp.dataPro):
    '''
    每次预测之前都去训练之前的数据
    '''
    def __init__(self, filePath=""):
        super(dp.dataPro).__init__()
        self.filePath = filePath

    def train(self,startDate, endDate=-1,preDays=5,futureDays=1):
        '''
        用startDate到 endDate的数据训练，预测futureDays天后的状态
        '''

        data = self.readData(self.filePath)
        if endDate == -1:
            endDate = len(data["open"]) - 1
        for date in range(startDate+preDays,endDate):
            train_x, train_y = self.getHistoryData(startDate,date,preDays,futureDays)
            svm_clf = svm.SVR()
            svm_clf.fit(train_x,train_y*100)
            x_ ,y_= self.getBatchData(date,preDays,futureDays)
            y_predict = svm_clf.predict([x_])
            print("y_true:{},\t y_preadict:{}".format(y_,y_predict))


def svmPredict_demo():
    # file = "./data/sh.600061.csv"
    root = "./data"
    files = os.listdir(root)
    
    for file in files:
        file = os.path.join(root,file)
        print("file:",file)
        data = dp.readData(file)
        sg = svmPredict(data)
        x_train,y_train,x_test,y_test = sg.getTrainData()
        svm_clf = svm.SVR()
        svm_clf.fit(x_train,y_train)
        print("svm_clf y_pred:{}, y:{}".format(svm_clf.predict([x_test[2]]),y_test[2]))
        print("clf score:", svm_clf.score(x_test, y_test))
        totalNum = 0
        accNum = 0
        for x, y in zip(x_test, y_test):
            y_p = svm_clf.predict([x])
            totalNum += 1
            if (y_p > 0 and y > 0) or (y_p < 0 and y < 0):
                accNum += 1
        print("totalNum:{},accNum:{}, acc: {}".format(totalNum, accNum, accNum/totalNum))



        svm_classifier = svm.SVC()
        y_test2 = np.array(y_test)
        y_train2 = np.array(y_train)
        y_train2 = np.where(y_train2>0,1,-1)
        y_test2 = np.where(y_test2>0,1,-1)
        svm_classifier.fit(x_train,y_train2)
        print("classifier score:",svm_classifier.score(x_test,y_test2))
        totalNum = 0
        accNum = 0
        for x,y in zip(x_test,y_test2):
            y_p = svm_classifier.predict([x])
            totalNum += 1
            if y_p == y:
                accNum += 1
        print("totalNum:{},accNum:{}".format(totalNum,accNum))

        print("svm_classifier y_pred:{}, y:{}".format(svm_classifier.predict([x_test[1]]),y_test2[1]))



def svmPredict_continuously_demo():
    file = "./data/sh.600061.csv"
    model = svmPredict_continuously(file)

    startIndex = 20
    model.train(startIndex)


if __name__ == "__main__":
    svmPredict_demo()
    #svmPredict_continuously_demo()