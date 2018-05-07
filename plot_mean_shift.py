"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""
print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from random import *

# #############################################################################
# Generate sample data
allerr = []
errs = []
wrongcluster = []
for h in range(50):
    del errs[:]
    ooops = 0
    for i in range(50):

        centers = [[randint(0, 1000), randint(0, 1000)], [randint(0, 1000), randint(0, 1000)], [randint(0, 1000), randint(0, 1000)]]
        X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

        # #############################################################################
        # Compute clustering with MeanShift

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        #print('actual')
        #print(centers)
        #print('------predicted-------')
        #print(cluster_centers)
        #print('------error-----')

        counter = 0
        if n_clusters_ != len(centers):
            counter = 13
            ooops = ooops + 1
            total_err = 0.2
        while (counter != 13):
            if counter == 0:
                diff1 = centers[0] - cluster_centers[0]
                diff2 = centers[1] - cluster_centers[1]
                diff3 = centers[2] - cluster_centers[2]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 1:
                diff1 = centers[0] - cluster_centers[2]
                diff2 = centers[1] - cluster_centers[0]
                diff3 = centers[2] - cluster_centers[1]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 2:
                diff1 = centers[0] - cluster_centers[1]
                diff2 = centers[1] - cluster_centers[2]
                diff3 = centers[2] - cluster_centers[0]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 3:
                diff1 = centers[2] - cluster_centers[0]
                diff2 = centers[0] - cluster_centers[1]
                diff3 = centers[1] - cluster_centers[2]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 4:
                diff1 = centers[1] - cluster_centers[0]
                diff2 = centers[2] - cluster_centers[1]
                diff3 = centers[0] - cluster_centers[2]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 5:
                diff1 = centers[0] - cluster_centers[2]
                diff2 = centers[1] - cluster_centers[1]
                diff3 = centers[2] - cluster_centers[0]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 6:
                diff1 = centers[0] - cluster_centers[1]
                diff2 = centers[1] - cluster_centers[0]
                diff3 = centers[2] - cluster_centers[2]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 7:
                diff1 = centers[0] - cluster_centers[0]
                diff2 = centers[1] - cluster_centers[2]
                diff3 = centers[2] - cluster_centers[1]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 8:
                diff1 = centers[2] - cluster_centers[1]
                diff2 = centers[0] - cluster_centers[0]
                diff3 = centers[1] - cluster_centers[2]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 9:
                diff1 = centers[2] - cluster_centers[1]
                diff2 = centers[0] - cluster_centers[2]
                diff3 = centers[1] - cluster_centers[0]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 10:
                diff1 = centers[2] - cluster_centers[0]
                diff2 = centers[0] - cluster_centers[2]
                diff3 = centers[1] - cluster_centers[1]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 11:
                diff1 = centers[2] - cluster_centers[2]
                diff2 = centers[0] - cluster_centers[0]
                diff3 = centers[1] - cluster_centers[1]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break
            elif counter == 12:
                diff1 = centers[2] - cluster_centers[2]
                diff2 = centers[0] - cluster_centers[1]
                diff3 = centers[1] - cluster_centers[0]
                error = (diff1**2 + diff2**2 + diff3**2)
                total_err = error[0] + error[1]
                #print('total error: %d' % total_err)
                if (total_err < 3.0):
                    break

            counter = counter + 1
        errs.append(total_err)
        #print(total_err)


        #print("number of estimated clusters : %d" % n_clusters_)
        #print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')


        # #############################################################################
        # Plot result
        import matplotlib.pyplot as plt
        from itertools import cycle

        #plt.figure(1)
        #plt.clf()

        #colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        #for k, col in zip(range(n_clusters_), colors):
            #my_members = labels == k
            #cluster_center = cluster_centers[k]
            #plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     #markeredgecolor='k', markersize=14)
        #plt.title('Estimated number of clusters: %d' % n_clusters_)
        #plt.show()

    #print('Length of error list:')
    #print(len(errs))
    #print('Length of failed cluster estimate:')
    #print(ooops)
    wrongcluster.append(ooops)
    ooops = 0
    #plt.plot(range(100), errs, 'b-')
    #plt.ylabel('mean squared error')
    #plt.xlabel('trial')
    #plt.show()

    #print('*-*-*-* STATS *-*-*-*')
    #print('Average error:')
    #print(sum(errs)/len(errs))
    allerr.append(sum(errs)/len(errs))
    #print('TOTAL WRONG CLUSTER EST:')
    #print(sum(wrongcluster))
print('On trial #: %d' % h)
#plt.plot(range(50), errs, 'b-')
#plt.show()
#plt.scatter(range(1,11),allerr)
#plt.show()
print(sum(wrongcluster))
#print(aller)

plt.clf()
plt.ylabel('mean squared error')
plt.xlabel('trials of 50 samples each with 1000 points equidivided among groups')
plt.errorbar(range(1, 51),allerr,yerr=allerr, fmt='o')
plt.show()
