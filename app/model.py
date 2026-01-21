"""
LDCT Denoising - Model Module
Generator model yükleme ve inference.
"""

import os
import numpy as np

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow loglarını azalt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None
    layers = None


# Model sabitleri
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 1


def downsample(filters, size, apply_batchnorm=True):
    """Encoder katmanı"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """Decoder katmanı"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result


def build_generator():
    """U-Net Generator mimarisini oluşturur"""
    inputs = layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, CHANNELS])
    
    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    
    # Decoder
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)


class LDCTModel:
    """LDCT Denoising Generator Model"""
    
    def __init__(self):
        self.generator = None
        self.is_loaded = False
        
    def load_weights(self, weights_path: str) -> bool:
        """
        Model ağırlıklarını yükler.
        
        Args:
            weights_path: .h5 dosya yolu
            
        Returns:
            bool: Başarılı ise True
        """
        if tf is None:
            raise ImportError("TensorFlow yüklü değil. 'pip install tensorflow' komutunu çalıştırın.")
        
        try:
            # Generator oluştur
            self.generator = build_generator()
            
            # Ağırlıkları yükle
            self.generator.load_weights(weights_path)
            self.is_loaded = True
            print(f"Model başarıyla yüklendi: {weights_path}")
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, input_image: np.ndarray) -> np.ndarray:
        """
        Görüntü üzerinde inference yapar.
        
        Args:
            input_image: (1, 256, 256, 1) shape'inde normalized input
            
        Returns:
            numpy array: (1, 256, 256, 1) shape'inde model çıktısı
        """
        if not self.is_loaded:
            raise RuntimeError("Model henüz yüklenmedi!")
        
        # Inference
        output = self.generator(input_image, training=False)
        
        return output.numpy()


# Global model instance
_model_instance = None


def get_model() -> LDCTModel:
    """Global model instance'ını döndürür"""
    global _model_instance
    if _model_instance is None:
        _model_instance = LDCTModel()
    return _model_instance


def load_model(weights_path: str) -> bool:
    """Model'i yükler"""
    model = get_model()
    return model.load_weights(weights_path)


def predict(input_image: np.ndarray) -> np.ndarray:
    """Inference yapar"""
    model = get_model()
    return model.predict(input_image)
