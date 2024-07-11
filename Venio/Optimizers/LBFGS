void LBFGS::updateWeights(double learning_speed, int m) {
    // Ограниченное хранилище для векторов s и y
    std::deque<Vector> s_list;
    std::deque<Vector> y_list;
    std::deque<double> ro_list;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 1; i < _network.getLayersSize(); i++) {
            _layer_weights_gradient = _network.getLayerWeightsGradient(i);
            _layer_weights = _network.getLayerWeights(i);

            _old_gradient[i] = Eigen::Map<Vector>(_layer_weights_gradient.data(),
                                                  _layer_weights_gradient.size());

            _old_weights[i] = Eigen::Map<Vector>(_layer_weights.data(),
                                                 _layer_weights.size());

            // Вычисление направления с использованием двухцикловой рекурсии
            Vector q = _old_gradient[i];
            std::vector<double> alpha(m);
            for (int j = s_list.size() - 1; j >= 0; --j) {
                alpha[j] = ro_list[j] * s_list[j].dot(q);
                q -= alpha[j] * y_list[j];
            }

            // Масштабирование начального приближения матрицы Гессе
            double gamma = s_list.back().dot(y_list.back()) / y_list.back().dot(y_list.back());
            q *= gamma;

            for (int j = 0; j < s_list.size(); ++j) {
                double beta = ro_list[j] * y_list[j].dot(q);
                q += s_list[j] * (alpha[j] - beta);
            }

            _new_weights[i] = _old_weights[i] - learning_speed * q;

            _network.setLayerWeights(i, Eigen::Map<Matrixd>(_new_weights[i].data(),
                                                            _layer_weights.rows(),
                                                            _layer_weights.cols()));
        }
        _network.forwardPropagation();

        // ПРЕДУПРЕЖДЕНИЕ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        _network.backPropagation();

        for (int i = 1; i < _network.getLayersSize(); i++) {
            _new_gradient[i] = Eigen::Map<Vector>(_network.getLayerWeightsGradient(i).data(),
                                                  _network.getLayerWeightsGradient(i).size());

            // Обновление s и y
            Vector s = _new_weights[i] - _old_weights[i];
            Vector y = _new_gradient[i] - _old_gradient[i];
            double ro = 1.0 / y.dot(s);

            // Хранить только последние m пар (s, y)
            if (s_list.size() >= m) {
                s_list.pop_front();
                y_list.pop_front();
                ro_list.pop_front();
            }

            s_list.push_back(s);
            y_list.push_back(y);
            ro_list.push_back(ro);
        }
    }
}
