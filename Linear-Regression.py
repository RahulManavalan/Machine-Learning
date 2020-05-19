{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled22.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOTsKW2YGqw0nTnDidyu2vx",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/RahulManavalan/Machine-Learning/blob/master/Linear-Regression.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7I7UL1ArC7-I",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "pip install tensorflow==1.0"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a3T990LnSBRt",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd \n",
        "import numpy as np \n",
        "from sklearn import datasets,linear_model\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib.pyplot import scatter,plot"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sqlX5a-2iOP3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Helper function to read Dara\n",
        "\n",
        "def read_goog_spm500_data(): \n",
        "  googFile = \"/content/GOOG.csv\"\n",
        "  spFile = \"/content/^GSPC.csv\"\n",
        "\n",
        "  goog = pd.read_csv(googFile,sep=\",\",usecols=[0,5],names=[\"Date\",\"goog\"],header=0) \n",
        "  sp = pd.read_csv(spFile,sep=\",\",usecols=[0,5],names=[\"Date\",\"SP500\"],header=0) \n",
        "  goog[\"SP500\"] = sp[\"SP500\"]\n",
        "  \n",
        "  goog[\"Date\"] = pd.to_datetime(goog[\"Date\"],format=\"%Y-%m-%d\")\n",
        "  goog = goog.sort_values([\"Date\"],ascending=True) \n",
        "\n",
        "  returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in [\"float64\",\"int64\"]]].pct_change() \n",
        "\n",
        "  xData = np.array(returns[\"SP500\"])[1:] \n",
        "  yData = np.array(returns[\"goog\"])[1:] \n",
        "\n",
        "  return(xData,yData) "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lQ6wcOOAi0jd",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 54
        },
        "outputId": "6e56ccce-fda5-4268-b5f7-25c4700f3df8"
      },
      "source": [
        "xData , yData = read_goog_spm500_data()\n",
        "googModel = linear_model.LinearRegression()\n",
        "googModel.fit(xData.reshape(-1,1) , yData.reshape(-1,1))\n",
        "\n",
        "print(googModel.coef_)\n",
        "print(googModel.intercept_)\n"
      ],
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[1.05921818]]\n",
            "[0.00792521]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KfOpc-S2i2v-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 719
        },
        "outputId": "f94e00a5-f29e-4442-8d3c-05ce90f35d0e"
      },
      "source": [
        "plt.figure(figsize=(10,6))\n",
        "scatter(xData,yData) \n"
      ],
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-81-eea253c09358>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m6\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mxData\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0myData\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgoogModel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mintercept_\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0mgoogModel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcoef_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/pyplot.py\u001b[0m in \u001b[0;36mplot\u001b[0;34m(scalex, scaley, data, *args, **kwargs)\u001b[0m\n\u001b[1;32m   2761\u001b[0m     return gca().plot(\n\u001b[1;32m   2762\u001b[0m         *args, scalex=scalex, scaley=scaley, **({\"data\": data} if data\n\u001b[0;32m-> 2763\u001b[0;31m         is not None else {}), **kwargs)\n\u001b[0m\u001b[1;32m   2764\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2765\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_axes.py\u001b[0m in \u001b[0;36mplot\u001b[0;34m(self, scalex, scaley, data, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1644\u001b[0m         \"\"\"\n\u001b[1;32m   1645\u001b[0m         \u001b[0mkwargs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcbook\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnormalize_kwargs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmlines\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mLine2D\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1646\u001b[0;31m         \u001b[0mlines\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_lines\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1647\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mline\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mlines\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1648\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd_line\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mline\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_base.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    214\u001b[0m                 \u001b[0mthis\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    215\u001b[0m                 \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 216\u001b[0;31m             \u001b[0;32myield\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_plot_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mthis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    217\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    218\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget_next_color\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_base.py\u001b[0m in \u001b[0;36m_plot_args\u001b[0;34m(self, tup, kwargs)\u001b[0m\n\u001b[1;32m    340\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    341\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 342\u001b[0;31m             raise ValueError(f\"x and y must have same first dimension, but \"\n\u001b[0m\u001b[1;32m    343\u001b[0m                              f\"have shapes {x.shape} and {y.shape}\")\n\u001b[1;32m    344\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m2\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: x and y must have same first dimension, but have shapes (100,) and (1, 100)"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x432 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fgwN0SHioXuL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}