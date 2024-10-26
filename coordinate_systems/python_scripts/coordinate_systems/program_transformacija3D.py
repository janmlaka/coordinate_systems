from coordinate_systems.python_scripts.coordinate_systems.proj2kart import proj2kart_fun, GK2FLh, DolMer
from coordinate_systems.python_scripts.coordinate_systems.kart2proj import kart2proj_fun, xyz2flh, FLh2GK
import numpy as np
import math
import os
import json

def transformacija_D48_D96(D48_file, D96_file):
    #faktorja za skeliranje
    faktorW = 180/math.pi * 3600
    faktorM = 1e6

    #parametri elipsoida bessel
    srednji_meridian_bessel = math.radians(15)
    m0_bessel = 0.9999
    a_bessel = 6377397.155
    e_bessel = 8.169683087477896e-02
    False_N_bessel = 5000000
    False_E_bessel = 500000

    #parametri elipsoida GRS80
    srednji_meridian_GRS80 = math.radians(15)
    m0_GRS80 = 0.9999
    a_GRS80 = 6378137.000
    e_GRS80 = 8.181919104281514e-02
    False_N_GRS80 = 5000000
    False_E_GRS80 = 500000

    #branje podatkov
    column1_48 = [] #X
    column2_48 = [] #Y
    column3_48 = [] #h
    column4_48 = [] #X
    column5_48 = [] #Y
    column6_48 = [] #h
    with open(D48_file, 'r') as koordinate_D48_file:
        for coordinates_48 in koordinate_D48_file:
            split_lines = coordinates_48.split()
            if len(split_lines) >= 6:
                column1_48.append(split_lines[0])
                column2_48.append(split_lines[1])
                column3_48.append(split_lines[2])
                column4_48.append(split_lines[3])
                column5_48.append(split_lines[4])
                column6_48.append(split_lines[5])

    column1_96 = [] #n
    column2_96 = [] #e
    column3_96 = [] #h
    column4_96 = [] #n
    column5_96 = [] #e
    column6_96 = [] #h

    with open(D96_file, 'r') as koordinate_D96_file:
        for coordinates_96 in koordinate_D96_file:
            split_lines = coordinates_96.split()
            if len(split_lines) >= 6:
                column1_96.append(split_lines[0])
                column2_96.append(split_lines[1])
                column3_96.append(split_lines[2])
                column4_96.append(split_lines[3])
                column5_96.append(split_lines[4])
                column6_96.append(split_lines[5])


    k = len(column1_48) 
    n = 6 * k
    u = 7
    n0 = u + 3*k
    r = n - n0
    c = r + u 

    kartezicne_D48 = np.zeros((k,3))
    la_fi_D48 = np.zeros((k,2))
    for i in range(k):
        kartezicne_coo = np.array([proj2kart_fun(column2_48[i], column1_48[i], column3_48[i], 
                    srednji_meridian_bessel, m0_bessel, a_bessel, e_bessel,
                        False_N_bessel, False_E_bessel)])
        
        la_fi = np.array([GK2FLh(column2_48[i], column1_48[i], 
                    srednji_meridian_bessel, m0_bessel, a_bessel, e_bessel,
                        False_N_bessel, False_E_bessel)])
        

        kartezicne_D48[i,:] = kartezicne_coo.flatten()
        la_fi_D48[i, :] = la_fi.flatten()


    kartezicne_D96 = np.zeros((k,3))
    la_fi_D96 = np.zeros((k,2))
    for i in range(k):
        kartezicne_coo =  np.array([proj2kart_fun(column2_96[i], column1_96[i], column3_96[i],
                    srednji_meridian_GRS80, m0_GRS80, a_GRS80, e_GRS80,
                        False_N_GRS80, False_E_GRS80)])
        
        la_fi = np.array([GK2FLh(column2_96[i], column1_96[i], 
                    srednji_meridian_GRS80, m0_GRS80, a_GRS80, e_GRS80,
                        False_N_GRS80, False_E_GRS80)])
        
        kartezicne_D96[i,:] = kartezicne_coo.flatten()
        la_fi_D96[i, :] = la_fi.flatten()

    #IZRAVNAVA
    tx = 0
    ty = 0
    tz = 0
    wx = 0
    wy = 0
    wz = 0
    m = 1

    X = np.array([tx, ty, tz, wx, wy, wz, m])
    X0 = np.array([tx, ty, tz, wx, wy, wz, m])
    odsList = []
    ni = 1
    ods = 1
    while ods > 1e-6:
        Rx = np.array([[1.0, 0.0, 0.0], 
                [0.0, np.cos(wx), -np.sin(wx)],
                [0.0, np.sin(wx), np.cos(wx)]]
                ,dtype=float)

        Ry = np.array([[np.cos(wy), 0.0, np.sin(wy)], 
                    [0.0, 1.0, 0.0],
                    [-np.sin(wy), 0.0, np.cos(wy)]]
                    ,dtype=float)
        
        Rz = np.array([[np.cos(wz), -np.sin(wz), 0.0],
                        [np.sin(wz), np.cos(wz), 0.0],
                    [0.0, 0.0, 1.0]]
                    ,dtype=float)
        
        R = Rx @ Ry @ Rz
        T = np.array([tx, ty, tz])
        f = np.zeros((c,1))
        index_i = 0
        for enacba in range(k):
            #print(kartezicne_D48[enacba])
            F = T.transpose() + (m * R @ (kartezicne_D48[enacba])) - (kartezicne_D96[enacba])
            index = index_i + (enacba) * 3
            index_end = index + 3
            f[index: index_end, 0:1] = F[:3].reshape((3,1))
        f = -f

        A = np.zeros((c,n))
        for odvod  in range(k):
            F_D48 = m*R
            F_D96 = -np.eye(3)
            element_1 = np.hstack((F_D48, F_D96))
            #print(element_1)
            element_A_1 = (odvod) * 3  
            element_A_2 = (odvod) * 6  
            A[element_A_1:element_A_1 + 3, element_A_2:element_A_2 + 6] = element_1
        #print("A: \n",A)
        #parcialni odvodi po neznankah
        # po rotacijah

        Rx_wx = np.array([[0, 0, 0], 
                        [0, -np.sin(X0[3]), -np.cos(X0[3])], 
                        [0, np.cos(X0[3]), -np.sin(X0[3])]], dtype='float64')
        
        Ry_wy = np.array([[-np.sin(X0[4]), 0, np.cos(X0[4])], 
                                        [0, 0, 0], 
                        [-np.cos(X0[4]), 0, -np.sin(X0[4])]], dtype='float64')
        
        Rz_wz = np.array([[-np.sin(X0[5]), -np.cos(X0[5]), 0], 
                        [np.cos(X0[5]), -np.sin(X0[5]), 0], 
                        [0, 0, 0]], dtype='float64')
        
        B = np.zeros((c, u))
        for neznanke in range(k):
            F_T = np.eye(3)
            F_wx = (m * Rx_wx @ Ry @ Rz @ (kartezicne_D48[neznanke][:, np.newaxis])) / faktorW
            F_wy = (m * Rx @ Ry_wy @ Rz @ (kartezicne_D48[neznanke][:, np.newaxis])) / faktorW
            F_wz = (m * Rx @ Ry @ Rz_wz @ (kartezicne_D48[neznanke][:, np.newaxis])) / faktorW
            F_m = (R @ (kartezicne_D48[neznanke][:, np.newaxis])) / faktorM
            element_B = np.hstack((F_T, F_wx, F_wy, F_wz, F_m))
            #print(element_B)
            element_B_1 = neznanke * 3
            element_B_2 = (neznanke + 1) * 3
            B[element_B_1: element_B_2, :] = element_B

        S = np.zeros((n,n))
        Si = np.zeros((n,n))
        S_G = np.zeros((6,6))
        Sr_48 = np.zeros((3,3))
        Sr_96 = np.zeros((3,3))
        for i in range(k):
            sigma_x48 = float(column4_48[i])
            sigma_y48 = float(column5_48[i])
            sigma_z48 = float(column6_48[i])

            sigma_x96 = float(column4_96[i])
            sigma_y96 = float(column5_96[i])
            sigma_z96 = float(column6_96[i])

            fi_d48_i = la_fi_D48[i,1]
            la_d48_i = la_fi_D48[i,0]
      
            fi_d96_i = la_fi_D96[i,1]
            la_d96_i = la_fi_D96[i,0]
            #variance
            Sr_48 = np.array([sigma_x48**2, sigma_y48**2, sigma_z48**2])
            Sr_96 = np.array([sigma_x96**2, sigma_y96**2, sigma_z96**2])
            #variance za posamezen k.s.
            Sr_D48_GK = np.zeros((3, 3))
            np.fill_diagonal(Sr_D48_GK, Sr_48)

            Sr_D96_TM = np.zeros((3, 3))
            np.fill_diagonal(Sr_D96_TM, Sr_96)

            J_D48 = np.array([[-math.sin(fi_d48_i)*math.cos(la_d48_i), -math.sin(la_d48_i), math.cos(fi_d48_i)*math.cos(la_d48_i)],
               [-math.sin(fi_d48_i)*math.sin(la_d48_i), math.cos(la_d48_i), math.cos(fi_d48_i)*math.sin(la_d48_i)],
                        [math.cos(fi_d48_i),        0,                math.sin(fi_d48_i)]])

            J_D96 = np.array([[-math.sin(fi_d96_i)*math.cos(la_d96_i), -math.sin(la_d96_i), math.cos(fi_d96_i)*math.cos(la_d96_i)],
               [-math.sin(fi_d96_i)*math.sin(la_d96_i), math.cos(la_d96_i), math.cos(fi_d96_i)*math.sin(la_d96_i)],
                        [math.cos(fi_d96_i),        0,                math.sin(fi_d96_i)]])

            S_G_D48 = J_D48 @ Sr_D48_GK @ J_D48.transpose()
            S_G_D96 = J_D96 @ Sr_D96_TM @ J_D96.transpose()

            S_G[0:3, 0:3] = S_G_D48
            S_G[3:6, 3:6] = S_G_D96

            start_row_Si = (i) * 6   # 0 5 11...
            end_row_Si = (i) * 6 + 6 #6 12 18...
            Si[start_row_Si:end_row_Si, start_row_Si:end_row_Si] = S_G
            S = Si
        
        var_apr = np.mean(np.mean(S))
        Q = (1/var_apr)*S
        P = np.linalg.inv(Q)
        Qe = A @ Q @ A.transpose()       
        Pe = np.linalg.inv(Qe)    
        N = B.transpose()@Pe@B  
        t = B.transpose()@Pe@f
        delta = np.linalg.inv(N)@t
        
        tx = X0[0] + delta[0]
        tx = tx[0]    
        ty = X0[1] + delta[1]
        ty = ty[0]
        tz = X0[2] + delta[2]
        tz = tz[0]
        
        wx = X0[3] + delta[3] / faktorW
        wx = wx[0]
        wy = X0[4] + delta[4] / faktorW
        wy = wy[0]
        wz = X0[5] + delta[5] / faktorW
        wz = wz[0]
        
        m = X0[6] + delta[6] / faktorM
        m = m[0]
        
        X0 = np.array([tx, ty, tz, wx, wy, wz, m])
        #print("X0: \n",X0)
        X = np.array([tx, ty, tz, wx, wy, wz, m])
        #print("X: \n",X)

        v = Q@A.transpose()@Pe@(f-B@delta)
        
        ods = (np.linalg.norm(delta))
        odsList.append(ods)

        

        ni += 1

        if ni > 15:
            break

    T = np.array([X[0], X[1], X[2]]).transpose()

    Rx = np.array([[1.0, 0.0, 0.0], 
                [0.0, np.cos(X0[3]), -np.sin(X0[3])],
                [0.0, np.sin(X0[3]), np.cos(X0[3])]], dtype=float)


    Ry = np.array([[np.cos(X0[4]), 0.0, np.sin(X0[4])], 
                    [0.0, 1.0, 0.0],
                    [-np.sin(X0[4]), 0.0, np.cos(X0[4])]], dtype=float)

    Rz = np.array([[np.cos(X0[5]), -np.sin(X0[5]), 0.0],
                    [np.sin(X0[5]), np.cos(X0[5]), 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)


    R = Rx @ Ry @ Rz

    m = X[6]

    #varianca in standartni odklon
    rvarI = (v.transpose()@P@v)/r
    rStDI = math.sqrt(rvarI)

    varI = rvarI[0][0]
    StDI = rStDI

    #natan?nost neznank
    Qdd = np.linalg.inv(N)
    #natan?nost popravkov
    Qvv = Q @ A.transpose() @ Pe @(np.eye(c) - B @ Qdd @ B.transpose() @ Pe)@ A @ Q
    #natan?nost izravnanih koli?in
    QLi = Q - Qvv

    #natancnost neznank - transformacijski parametri
    Sdd = rvarI * Qdd

    Std_x = (math.sqrt(Sdd[0,0]))
    Std_y = (math.sqrt(Sdd[1,1]))
    Std_z = (math.sqrt(Sdd[2,2]))
    Std_wx = (math.sqrt(Sdd[3,3]))
    Std_wy = (math.sqrt(Sdd[4,4]))
    Std_wz = (math.sqrt(Sdd[5,5]))
    Std_m = (math.sqrt(Sdd[6,6]))

    Svv = rvarI * Qvv
    SLi = rvarI * QLi

    #PRENOS VAR IN KOVAR
    #COVARIANCE MATRIX
    D48_acc_mat = np.zeros((k*3, k*3))
    D48_acc_li = []
    for i in range(k):
        std_X = column4_48[i]
        std_Y = column5_48[i]
        std_h = column6_48[i]
        D48_acc_li.extend([std_X, std_Y, std_h])
    

    np.fill_diagonal(D48_acc_mat, D48_acc_li) 

    D48_coo = D48_acc_mat   

    # Compute the covariance matrix of the differences - 46x46
    cov_matrix = np.block([
        [Sdd, np.zeros((7,k*3))],
        [np.zeros((k*3,7)), D48_coo]
    ])

    #Assemble Jacobian matrix
    s = 3*k + 7
    J = np.zeros((k*3, s)) #39x46

    D48_J = np.zeros((k,3),dtype='float64')
    for i in range(k):
        X_D48 = column1_48[i]
        Y_D48 = column2_48[i]
        h_D48 = column3_48[i]
        ele_D48_coo = np.hstack([X_D48, Y_D48, h_D48])
        D48_J[i] = ele_D48_coo

    ele_D48 = m * R #3x3
    for i in range(k):
        if i * 3 + 3 <= k*3:
            J[i * 3:i * 3 + 3, i * 3:i * 3 + 3] = ele_D48

    
    ele_T = np.eye(3)
    ele_wx = (m * Rx_wx @ Ry @ Rz @ D48_J[i].reshape(-1)).reshape(3,1)/faktorW
    ele_wy = (m * Rx @ Ry_wy @ Rz @ D48_J[i].reshape(-1)).reshape(3,1)/faktorW
    ele_wz = (m * Rx @ Ry @ Rz_wz @ D48_J[i].reshape(-1)).reshape(3,1)/faktorW
    ele_m = (R @ D48_J[i].reshape(-1)).reshape(3,1)/faktorM
    ele_nezn = np.hstack([ele_T, ele_wx, ele_wy, ele_wz, ele_m])

    # Ensure we place it in a valid part of J
    row_start = 3 * (k - 1)  # Adjusted to place in the second-to-last block
    col_start = k * 3  # Adjusted to ensure valid indexing

    if row_start + 3 <= k * 3 and col_start + 7 <= s:
        J[row_start:row_start + 3, col_start:col_start + 7] = ele_nezn

    Sy = J @ cov_matrix @ J.T

    rezultati_txt = "rezultati.txt"

    with open(rezultati_txt, 'w') as rezultat:
        rezultat.write("REPORT OF THE RESULTS OF THE TRANSFORMATION BETWEEN D48 AND D96\n")
        rezultat.write("\nD48 - file: \n")
        rezultat.write("\n" + os.path.basename(D48_file) + "\n")
        rezultat.write("\nD96 - file: \n")
        rezultat.write("\n" + os.path.basename(D96_file) + "\n")
        rezultat.write("\nINPUT DATA - PROJECTION COORDINATES - D48/GK [m]\n")
        rezultat.write("     Y[m]	    X[m]            h[m]         st_Y[m]       st_X[m]       st_h[m]\n")
        for i in range(len(column1_48)): 
            rezultat.write("{0:12.3f}".format(float(column2_48[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column1_48[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column3_48[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column4_48[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column5_48[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column6_48[i])) + "\n")

        rezultat.write("\nINPUT DATA - PROJECTION COORDINATES - D96-17/TM [m]\n")
        rezultat.write("     e[m]	    n[m]            h[m]         st_e[m]       st_n[m]       st_h[m]\n")
        for i in range(len(column1_96)): 
            rezultat.write("{0:12.3f}".format(float(column2_96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column1_96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column3_96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column4_96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column5_96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(column6_96[i])) + "\n")

        
        rezultat.write("\nDATA - CARTESIAN COORDINATES - D48/GK [m]\n")
        rezultat.write("	y[m]	 	x[m]		z[m]\n")
        kartezicne_D48 = (kartezicne_D48.tolist())
        for i in range(len(kartezicne_D48)):
            for e in range(3):
                urejene_D48_kart = str("{0:15.4f}".format(float(kartezicne_D48[i][e])))
                if e == 0:
                    rezultat.write(urejene_D48_kart + " ")
                elif e == 1:
                    rezultat.write(urejene_D48_kart + " ")
                else:
                    rezultat.write(urejene_D48_kart + "\n")

        rezultat.write("\nDATA - CARTESIAN COORDINATES - D96/TM [m]\n")
        rezultat.write("	y[m]	 	x[m]		z[m]\n")
        kartezicne_D96 = (kartezicne_D96.tolist())
        for i in range(len(kartezicne_D96)):
            for e in range(3):
                urejene_D96_kart = str("{0:15.4f}".format(float(kartezicne_D96[i][e])))
                if e == 0:
                    rezultat.write(urejene_D96_kart + " ")
                elif e == 1:
                    rezultat.write(urejene_D96_kart + " ")
                else:
                    rezultat.write(urejene_D96_kart + "\n")

        rezultat.write("\nnumber of iterations:  " + str(ni-1) + "\n")
        rezultat.write("\nVARIANCE AND STANDART DEVIATION - Variance A-priori\n")
        rezultat.write("Reference variance a-priori = " + str(("{0:10.6f}".format(float(var_apr)))) + "\n")
        rezultat.write("\nVARIANCE AND STANDART DEVIATION - Variance A-posteriori\n")
        rezultat.write("Reference variance a-posteriori = " + str(("{0:10.6f}".format(float(varI)))) + "\n")
        rezultat.write("Reference standart deviation a-posteriori = " + str(("{0:10.6f}".format(float(StDI)))) + "\n")

        # rezultat.write("\n Vector of corrections for each point: \n")
        # for correction in range(k):
        #     rezultat.write(str(correction+1)+ "\t" + str(np.round(v[correction][0],6)) + "\n")
        
        rezultat.write("\n Norm of vector of corrections through each iteration: \n")
        for i in range(len(odsList)):
            rezultat.write(str(i+1) + "\t" + "{0:5.5e}".format(float(odsList[i])) + "\n")

        rezultat.write("\nTRANSFORMATION PARAMETERS\n")
        wx_sec = (wx*3600)
        wy_sec = (wy*3600)
        wz_sec = (wz*3600)
        rezultat.write("Tx=\t" + str("{0:5.3f}".format(float(tx))) + " m" + "\n")
        rezultat.write("Ty=\t" + str("{0:5.3f}".format(float(ty))) + " m" + "\n")
        rezultat.write("Tz=\t" + str("{0:5.3f}".format(float(tz))) + " m" + "\n")
        rezultat.write("Wx=\t" + str("{0:9.6f}".format(float(wx_sec))) + " sec" + "\n")
        rezultat.write("Wy=\t" + str("{0:9.6f}".format(float(wy_sec))) + " sec" + "\n")
        rezultat.write("Wz=\t" + str("{0:9.6f}".format(float(wz_sec))) + " sec" + "\n")
        rezultat.write("m=\t" + str("{0:9.6f}".format(float(m))) + " ppm" + "\n")
        
        

        rezultat.write("Tx=\t" + str("{0:10.6f}".format(float(Std_x))) + " m" + "\n")
        rezultat.write("Ty=\t" + str("{0:10.6f}".format(float(Std_y))) + " m" + "\n")
        rezultat.write("Tz=\t" + str("{0:10.6f}".format(float(Std_z))) + " m" + "\n")
        rezultat.write("Wx=\t" + str("{0:10.6f}".format(float(Std_wx))) + " sec" + "\n")
        rezultat.write("Wy=\t" + str("{0:10.6f}".format(float(Std_wy))) + " sec" + "\n")
        rezultat.write("Wz=\t" + str("{0:10.6f}".format(float(Std_wz))) + " sec" + "\n")
        rezultat.write("m=\t" + str("{0:10.6f}".format(float(Std_m))) + " ppm" + "\n")
        
        n_96 = np.array(column1_96, dtype=float)
        e_96 = np.array(column2_96, dtype=float)
        H_96 = np.array(column3_96, dtype=float)

        proj_arr = np.zeros((k, 3))
        rezultat.write("\nTransformed Coordinates - D96-17/TM \n")
        rezultat.write("     		    e[m]	   n[m]	   	  h[m]\n")
        for tocka_D96 in range(k):
            tocke_D96 = T + m * R @ (kartezicne_D48[tocka_D96])
            projekcijske_coo = np.array([kart2proj_fun(tocke_D96[0], tocke_D96[1], tocke_D96[2],
                                        srednji_meridian_GRS80, m0_GRS80, a_GRS80, e_GRS80,
                                        False_N_GRS80, False_E_GRS80)])
            projekcijske_coo = np.round(projekcijske_coo, 3)
            proj_arr[tocka_D96, :] = projekcijske_coo.flatten()
            numbers = ["{0:12.3f}".format(float(num)) for num in projekcijske_coo[0]]
            rezultat.write(f"Point {tocka_D96+1} " + "\t")
            rezultat.write(' '.join(numbers) + "\n")

        rezultat.write("\nDifferences between Entry data and Transformed Coordinates - D96-17/TM and their standart deviations\n")
        #rezultat.write("\nThe standart deviations are calcuted based on subtraction of every difference from the average of the differences between Entry Data and Tranformed coordinates.\n")
        rezultat.write("        e[m]          n[m]         h[m]        std_e[m]      	std_n[m]         std_h[m]\n")
        differences = np.zeros((k,3))

        D48_coo = np.zeros((k*3, k*3))

        for tocka_D96 in range(k):
            difference_e_D96 = e_96 - proj_arr[:,0]
            difference_n_D96 = n_96 -  proj_arr[:,1]
            difference_H_D96 = H_96 - proj_arr[:,2]

            arr_diff = np.vstack([difference_e_D96, difference_n_D96, difference_H_D96])

            arr_diff = arr_diff.T

        for i in range(k):  # Loop over each variable
            rezultat.write("{0:12.3f}".format(float(difference_e_D96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(difference_n_D96[i])) + "  ")
            rezultat.write("{0:12.3f}".format(float(difference_H_D96[i])) + "  ")

            # Extract and calculate standard deviations from diagonal elements for e, n, and H
            k = Sy.shape[0]  # Determine the value of k based on the shape of Sy
            for j in range(3):  # Loop over the three components: e, n, H
                index = 3 * i + j  # Calculate the correct diagonal index for each component
                var = Sy[index][index]  # Extract the variance from the diagonal
                std = math.sqrt(var) # Calculate the standard deviation
                urejene_std = "{0:15.6f}".format(float(std))  # Format the standard deviation value
                rezultat.write(urejene_std + "  ")

            rezultat.write("\n")  # New line after each variable's output


        

    data = {
    "REPORT_OF_RESULTS": "TRANSFORMATION BETWEEN D48 AND D96",
    "D48_file": os.path.basename(D48_file),
    "D96_file": os.path.basename(D96_file),
    "INPUT_DATA_D48_GK": {
        "Y[m]": column2_48,
        "X[m]": column1_48,
        "h[m]": column3_48,
        "st_Y[m]": column4_48,
        "st_X[m]": column5_48,
        "st_h[m]": column6_48
    },
    "INPUT_DATA_D96_17_TM": {
        "e[m]": column2_96,
        "n[m]": column1_96,
        "h[m]": column3_96,
        "st_e[m]": column4_96,
        "st_n[m]": column5_96,
        "st_h[m]": column6_96
    },
    "DATA_CARTESIAN_D48_GK": kartezicne_D48,
    "DATA_CARTESIAN_D96_TM": kartezicne_D96,
    "number_of_iterations": ni - 1,
    "VARIANCE_AND_STANDART_DEVIATION": {
        "Variance_A-priori": float(var_apr),
        "Variance_A-posteriori": float(varI),
        "Standard_deviation_A-posteriori": float(StDI),
        "Norm_of_vector_of_corrections": {str(i + 1): float(odsList[i]) for i in range(len(odsList))}
    },
    "TRANSFORMATION_PARAMETERS": {
        "Tx": float(tx),
        "Ty": float(ty),
        "Tz": float(tz),
        "Wx": float(wx * 3600),
        "Wy": float(wy * 3600),
        "Wz": float(wz * 3600),
        "m": float(m)
    },
    "STANDART_DEVIATIONS_FOR_TRANSFORMATION_PARAMETERS": {
        "Tx": float(Std_x),
        "Ty": float(Std_y),
        "Tz": float(Std_z),
        "Wx": float(Std_wx),
        "Wy": float(Std_wy),
        "Wz": float(Std_wz),
        "m": float(Std_m)
    },
    "Transformed_Coordinates_D96_TM": {
        "e[m]": [proj[0] for proj in proj_arr],
        "n[m]": [proj[1] for proj in proj_arr],
        "h[m]": [proj[2] for proj in proj_arr]
    },
    "Differences_between_Entry_data_and_Transformed_Coordinates_D96_TM": {
        "e[m]": differences[:, 0].tolist(),
        "n[m]": differences[:, 1].tolist(),
        "h[m]": differences[:, 2].tolist(),
    }
}

    # Convert to JSON
    json_data = json.dumps(data, indent=4)

    # Write to file if needed
    with open('rezultati.json', 'w') as json_file:
        json_file.write(json_data)
