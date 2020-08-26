import pandas as pd
import numpy as np

#dc_analysis function has only one parameter: netlist file location as string
#It returns 6 outputs according to the relation AX = Z
# A, X and Z are returned in symbolic and numerical form
def dc_analysis(input_file):

    lst1 = ['Element', 'node1', 'node2', 'value']
    lst2 = ['Element', 'n+', 'n-', 'nc+', 'nc-', 'value']
    lst3 = ['Element', 'n+', 'n-', 'source', 'value']

    net_file = pd.read_table(input_file, header=None)

    indPass = []
    depVC = []
    depCC = []
    nothing = []

    #The loop reads the netlist file and separate them into 4 groups
    #1)lines that will be ignored as comments and type of analysis
    #2)lines describing resistors and independent sources
    #3)lines describing voltage controlled sources
    #4)lines describing current controlled sources
    for n in range(len(net_file)):
        k = str(net_file.iloc[n,0])
        if k[0] == 'V' or k[0] == 'v' or k[0] == 'R' or k[0] == 'r' or k[0] == 'I' or k[0] == 'i':
            indPass.append(n)
        elif k[0] == 'E' or k[0] == 'e' or k[0] == 'G' or k[0] == 'e':
           depVC.append(n)
        elif k[0] == 'H' or k[0] == 'h' or k[0] == 'F' or k[0] == 'f':
            depCC.append(n)
        else:
            nothing.append(n)

    #put the lines in their specified groups
    indPass_net = pd.read_table(input_file, sep = ' ', header = None, names = lst1, skiprows=depVC+depCC+nothing)
    depVC_net = pd.read_table(input_file, sep = ' ', header = None, names = lst2, skiprows=indPass+depCC+nothing)
    depCC_net = pd.read_table(input_file, sep = ' ', header = None, names = lst3, skiprows=indPass+depVC+nothing)

    elements1 = indPass_net.iloc[:,0]

    voltage_source = pd.DataFrame(data = None)
    resistor = pd.DataFrame(data = None)
    current_source = pd.DataFrame(data = None)

    num = 0

    #Do for loop to group resistors, current and voltage sources
    #We get three groups will be used to create matrices
    for i in elements1:
        if i[0] == 'V' or i[0] == 'v':
            voltage_source = voltage_source.append(indPass_net[num:num+1])
        elif i[0] == 'R' or i[0] == 'r':
            resistor = resistor.append(indPass_net[num:num+1])
        elif i[0] == 'I' or i[0] == 'i':
            current_source = current_source.append(indPass_net[num:num+1])

        num = num + 1


    num1 = 0
    num2 = 0

    elements2 = depVC_net.iloc[:,0]
    elements3 = depCC_net.iloc[:,0]

    VC_CS = pd.DataFrame(data = None)
    VC_VS = pd.DataFrame(data = None)
    CC_CS = pd.DataFrame(data = None)
    CC_VS = pd.DataFrame(data = None)

    for l in elements2:
        if l[0] == 'G' or l[0] == 'g':
            VC_CS = VC_CS.append(depVC_net[num1:num1+1])
        elif l[0] == 'E' or l[0] == 'e':
            VC_VS = VC_VS.append(depVC_net[num1:num1+1])

        num1 = num1 + 1


    for w in elements3:
        if w[0] == 'F' or w[0] == 'f':
            CC_CS = CC_CS.append(depCC_net[num2:num2 + 1])
        elif w[0] == 'H' or w[0] == 'h':
            CC_VS = CC_VS.append(depCC_net[num2:num2 + 1])

        num2 = num2 + 1

    numOfNodes = max(indPass_net['node1']) ####

    numOfVoltageS = len(voltage_source) + len(VC_VS) + len(CC_VS)

    numOfCurrentS = len(current_source)

    G1 = np.zeros([numOfNodes+1, numOfNodes+1])

    node1 = 0
    node2 = 0

    G_sym1 = [['' for i in range(numOfNodes+1)]for i in range(numOfNodes+1)]


    #G1 is the matrix of conductance
    for res_i in range(len(resistor)):
        node1 = resistor.iloc[res_i, 1]
        node2 = resistor.iloc[res_i, 2]

        G1[node1, node1] = G1[node1, node1] + 1 / resistor.iloc[res_i, 3]
        G1[node2, node2] = G1[node2, node2] + 1 / resistor.iloc[res_i, 3]
        G1[node1, node2] = G1[node1, node2] - 1 / resistor.iloc[res_i, 3]
        G1[node2, node1] = G1[node2, node1] - 1 / resistor.iloc[res_i, 3]
        G_sym1[node1][node1] = G_sym1[node1][node1] + '+1/' + resistor.iloc[res_i, 0]
        G_sym1[node2][node2] = G_sym1[node2][node2] + '+1/' + resistor.iloc[res_i, 0]
        G_sym1[node1][node2] = G_sym1[node1][node2] + '-1/' + resistor.iloc[res_i, 0]
        G_sym1[node2][node1] = G_sym1[node2][node1] + '-1/' + resistor.iloc[res_i, 0]



    nPos = 0
    nNeg = 0
    ncPos = 0
    ncNeg = 0

    #Modify G Matrix for /VC_CS/
    for VC_CSi in range(len(VC_CS)):
        nPos = VC_CS.iloc[VC_CSi,1]
        nNeg = VC_CS.iloc[VC_CSi,2]
        ncPos = VC_CS.iloc[VC_CSi,3]
        ncNeg = VC_CS.iloc[VC_CSi,4]

        G1[nPos, ncPos] = G1[nPos, ncPos] + VC_CS.iloc[VC_CSi, 5]
        G1[nNeg, ncNeg] = G1[nNeg, ncNeg] + VC_CS.iloc[VC_CSi, 5]
        G1[nPos, ncNeg] = G1[nPos, ncNeg] - VC_CS.iloc[VC_CSi, 5]
        G1[nNeg, ncPos] = G1[nNeg, ncPos] - VC_CS.iloc[VC_CSi, 5]
        G_sym1[nPos][ncPos] = G_sym1[nPos][ncPos]+'+'+VC_CS.iloc[VC_CSi, 0]
        G_sym1[nNeg][ncNeg] = G_sym1[nNeg][ncNeg]+'+'+VC_CS.iloc[VC_CSi, 0]
        G_sym1[nPos][ncNeg] = G_sym1[nPos][ncNeg]+'-'+VC_CS.iloc[VC_CSi, 0]
        G_sym1[nNeg][ncPos] = G_sym1[nNeg][ncPos]+'-'+VC_CS.iloc[VC_CSi, 0]

    G = G1[1:len(G1), 1:len(G1)]

    #print(G)
    B = np.zeros([numOfNodes,numOfVoltageS])
    B_sym = [['' for i in range(numOfVoltageS)]for i in range(numOfNodes)]

    D = np.zeros([numOfVoltageS, numOfVoltageS])
    D_sym = [['' for i in range(numOfVoltageS)]for i in range(numOfVoltageS)]

    #B is a matrix of voltage source
    for vs_i in range(len(voltage_source)):
        node1 = voltage_source.iloc[vs_i,1]
        node2 = voltage_source.iloc[vs_i,2]


        if node1 != 0:
            B[node1-1,vs_i] = 1
            B_sym[node1-1][vs_i] = '1'

        if node2 != 0:
            B[node2-1,vs_i] = -1
            B_sym[node2-1][vs_i] = '-1'

    #Modify B Matrix for /VC_VS/
    for VC_VSi in range(len(VC_VS)):

        nPos = VC_VS.iloc[VC_VSi, 1]
        nNeg = VC_VS.iloc[VC_VSi, 2]

        if nPos != 0:
            B[nPos - 1, VC_VSi + len(voltage_source)] = 1
            B_sym[nPos - 1][VC_VSi + len(voltage_source)] = '1'

        if nNeg != 0:
            B[nNeg - 1, VC_VSi + len(voltage_source)] = -1
            B_sym[nNeg - 1][VC_VSi + len(voltage_source)] = '-1'


    #Modify D and B Matrices for CC_VS
    for CC_VSi in range(len(CC_VS)):

        nPos = CC_VS.iloc[CC_VSi, 1]
        nNeg = CC_VS.iloc[CC_VSi, 2]

        if nPos != 0:
            B[nPos - 1, CC_VSi + len(voltage_source) + len(VC_VS)] = 1
            B_sym[nPos - 1][CC_VSi + len(voltage_source) + len(VC_VS)] = '1'

        if nNeg != 0:
            B[nNeg - 1, CC_VSi + len(voltage_source) + len(VC_VS)] = -1
            B_sym[nNeg - 1][CC_VSi + len(voltage_source) + len(VC_VS)] = '-1'

        for p in range(len(voltage_source)):
            if voltage_source.iloc[p,0] == CC_VS.iloc[CC_VSi,3]:
                D[len(voltage_source)+len(VC_VS)+CC_VSi, p] = -CC_VS.iloc[CC_VSi,4]
                D_sym[len(voltage_source) + len(VC_VS) + CC_VSi][p] = '-' + CC_VS.iloc[CC_VSi, 0]



    C = B.copy()
    C = C.transpose()

    C_sym1 = np.array(B_sym)
    C_sym = C_sym1.copy()
    C_sym = C_sym.transpose()




    #Modify C Matrix for /VC_VS/

    for VC_VSi in range(len(VC_VS)):
        ncPos = VC_VS.iloc[VC_VSi, 3]
        ncNeg = VC_VS.iloc[VC_VSi, 4]

        if ncPos != 0:
            C[VC_VSi + len(voltage_source), ncPos - 1] -= VC_VS.iloc[VC_VSi, 5]
            C_sym[VC_VSi + len(voltage_source)][ncPos - 1] = C_sym[VC_VSi + len(voltage_source)][ncPos - 1] + '-' + VC_VS.iloc[VC_VSi, 0]

        if ncNeg != 0:
            C[VC_VSi + len(voltage_source), ncNeg - 1] += VC_VS.iloc[VC_VSi, 5]
            C_sym[VC_VSi + len(voltage_source)][ncNeg - 1] = C_sym[VC_VSi + len(voltage_source)][ncNeg - 1] + '+' + VC_VS.iloc[VC_VSi, 0]

    #Modify B Matrix for /CC_CS/
    for CC_CSi in range(len(CC_CS)):
        nPos = CC_CS.iloc[CC_CSi, 1]
        nNeg = CC_CS.iloc[CC_CSi, 2]

        for t in range(len(voltage_source)):
            if voltage_source.iloc[t, 0] == CC_CS.iloc[CC_CSi, 3]:
                if nPos != 0:
                    B[nPos-1, t] += CC_CS.iloc[CC_CSi, 4]
                    B_sym[nPos - 1][t] = B_sym[nPos - 1][t] + '+' + CC_CS.iloc[CC_CSi, 0]

                if nNeg != 0:
                    B[nNeg - 1, t] -= CC_CS.iloc[CC_CSi, 4]
                    B_sym[nNeg - 1][t] = B_sym[nNeg - 1][t] + '-' + CC_CS.iloc[CC_CSi, 0]


    A1 = np.concatenate((G,B),axis=1)
    A2 = np.concatenate((C,D),axis=1)

    #Concatenate G, B, C and D matrices to form A
    A = np.concatenate((A1,A2))

    G_sym = np.array(G_sym1)
    G_sym = G_sym[1:len(G_sym), 1:len(G_sym)]

    B_sym = np.array(B_sym)
    C_sym = np.array(C_sym)
    D_sym = np.array(D_sym)

    A1_sym = np.concatenate((G_sym,B_sym),axis=1)
    A2_sym = np.concatenate((C_sym,D_sym),axis=1)

    A_sym = np.concatenate((A1_sym,A2_sym))

    Z = np.zeros([numOfNodes+numOfVoltageS,1])



    X_sym = [['']for i in range(numOfNodes+numOfVoltageS)]

    for c1 in range(numOfNodes):
        X_sym[c1][0] = 'Node ' + str(c1+1)

    for c2 in range(len(voltage_source)):
        X_sym[c2+numOfNodes][0] = voltage_source.iloc[c2,0] + '_Current'

    for c3 in range(len(VC_VS)):
        X_sym[c3 + numOfNodes + len(voltage_source)][0] = VC_VS.iloc[c3, 0] + '_Current'

    for c4 in range(len(CC_VS)):
        X_sym[c4 + numOfNodes + len(voltage_source) + len(VC_VS)][0] = CC_VS.iloc[c4, 0] + '_Current'



    Z_sym = [['']for i in range(numOfNodes+numOfVoltageS)]

    for k in range(numOfNodes,len(voltage_source)+numOfNodes,1):
        Z[k] = voltage_source.iloc[k-numOfNodes,3]
        Z_sym[k][0] = voltage_source.iloc[k-numOfNodes,0]


    for is_i in range(numOfCurrentS):
        node1 = current_source.iloc[is_i,1]
        node2 = current_source.iloc[is_i,2]
        if node1 != 0:
            Z[node1-1] = Z[node1-1] - current_source.iloc[is_i, 3]
            Z_sym[node1-1][0] = Z_sym[node1-1][0] +'-'+ current_source.iloc[is_i, 0]

        if node2 != 0:
            Z[node2-1] = Z[node2-1] + current_source.iloc[is_i, 3]
            Z_sym[node2-1][0] = Z_sym[node2-1][0] +'+'+ current_source.iloc[is_i, 0]

    A_inv = np.linalg.inv(A)
    X = A_inv.dot(Z)

    def put_zero(func):
        [d1, d2] = func.shape
        for m1 in range(d1):
            for m2 in range(d2):
                if func[m1,m2] == '':
                    func[m1,m2] = '0'

        return func


    Z_sym = np.array(Z_sym)
    Z_sym = put_zero(Z_sym)

    X_sym = np.array(X_sym)

    A_sym = put_zero(A_sym)

    return A, X, Z, A_sym, X_sym, Z_sym



