package smile

import smile.regression.GradientTreeBoost
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat

fun main() {
    // Чтение данных из CSV с использованием библиотеки Apache Commons CSV
    val dsFileFormat = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .setDelimiter(',')
        .build()

    // Чтение данных Advertising (обновите путь к файлу)
    val dataset = Read.csv("src/main/resources/Advertising.csv", dsFileFormat)

    // Выводим структуру данных для проверки
    println(dataset)

    // Формула для предсказания Sales (целевой переменной)
    val formula = Formula.lhs("sales")

    // Кросс-валидация для регрессии
    val res = CrossValidation.regression(
        10, formula, dataset,
        { formula, data -> GradientTreeBoost.fit(formula, data) }
    )

    println(res)
}




