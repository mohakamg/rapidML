class Overlayer:

    @staticmethod
    def overlay_configs(default_config, user_config):
        """
        Calls update keys externally to allow it to run recursively.
        :param default_config:
        :param user_config:
        :return user_config with updated keys:
        """
        return Overlayer.update_keys(default_config, user_config)

    @staticmethod
    def update_keys(default_dict, user_dict):
        """
        Recursively goes all the way down to the last values of each
        key in the default dictionary making sure that they all exist
        in the user dictionary while preserving the existing values
        in the user dictionary.
        :param default_dict:
        :param user_dict:
        :return: user_dict:
        """

        # check if default_dict is
        is_dictionary = isinstance(default_dict, dict)

        # if default_dict is a dictionary, there are still more dicts and/or values to be validated by default_dict
        if is_dictionary:
            for key in default_dict.keys():
                # if the current key is not in the user's dictionary,
                if key not in user_dict.keys():
                    # add the missing key from default_dict over to the user_config
                    user_dict[key] = default_dict.get(key)
                else:
                    # if the key exists in user_config validate it's children by recursively calling update_keys
                    # with the dictionary's children
                    user_dict[key] = Overlayer.update_keys(default_dict[key], user_dict[key])

        return user_dict